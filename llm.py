import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import warnings
import os
import pickle
import json
import argparse
from pathlib import Path
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Experiment metadata
    experiment_name: str = "default"
    experiment_type: str = "single"  # single, optimizer_comparison, architecture_ablation, scaling_study
    output_dir: str = "experiments"
    save_models: bool = False
    plot_results: bool = True
    
    # Experiment parameters
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    num_runs: int = 3
    
    # Specific experiment configs
    optimizer_variants: List[str] = field(default_factory=lambda: ["muon_hybrid", "adamw", "sgd"])
    architecture_variants: List[str] = field(default_factory=lambda: ["rmsnorm", "layernorm"])
    model_sizes: List[str] = field(default_factory=lambda: ["tiny", "small", "medium"])
    
@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 2000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    adamw_lr: float = 0.001
    sgd_lr: float = 0.01
    sgd_momentum: float = 0.9

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    
    # Experimental parameters
    optimizer_type: str = "muon_hybrid"  # muon_hybrid, adamw, sgd, adam
    norm_type: str = "rmsnorm"  # rmsnorm, layernorm
    pos_encoding: str = "rope"  # rope, sinusoidal, learned, none
    use_weight_tying: bool = True
    ns_steps: int = 5  # Newton-Schulz steps for Muon
    lr_schedule: str = "cosine_warmup"  # cosine_warmup, linear, step, constant
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @classmethod
    def get_model_size_config(cls, size: str) -> 'ModelConfig':
        """Get predefined model configurations"""
        configs = {
            "tiny": cls(d_model=192, n_heads=4, n_layers=4, d_ff=768, batch_size=32),
            "small": cls(d_model=384, n_heads=8, n_layers=6, d_ff=1536, batch_size=24),
            "medium": cls(d_model=512, n_heads=8, n_layers=8, d_ff=2048, batch_size=16),
            "large": cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072, batch_size=12)
        }
        return configs.get(size, configs["small"])

# Enhanced Newton-Schulz with adaptive steps
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class AdaptiveMuon(torch.optim.Optimizer):
    """Enhanced Muon with adaptive Newton-Schulz steps"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, 
                 ns_steps=5, adaptive_ns=False, min_ns_steps=1, max_ns_steps=10):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       ns_steps=ns_steps, adaptive_ns=adaptive_ns,
                       min_ns_steps=min_ns_steps, max_ns_steps=max_ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["step"] = 0

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                
                # Adaptive Newton-Schulz steps
                ns_steps = group["ns_steps"]
                if group["adaptive_ns"]:
                    grad_norm = g.norm().item()
                    # Reduce steps as gradients become smaller (more converged)
                    if grad_norm < 0.1:
                        ns_steps = max(group["min_ns_steps"], ns_steps - 2)
                    elif grad_norm > 1.0:
                        ns_steps = min(group["max_ns_steps"], ns_steps + 1)
                
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                
                state["step"] += 1

class Muon(torch.optim.Optimizer):
    """Original Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class SinusoidalPositionalEncoding(nn.Module):
    """Traditional sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, pos_encoding: str = "rope"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_encoding = pos_encoding

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        if pos_encoding == "rope":
            self.rotary = Rotary(self.d_k, max_seq_len)
        elif pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Apply positional encoding if not RoPE
        if self.pos_encoding == "sinusoidal":
            x = self.pos_enc(x)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if specified
        if self.pos_encoding == "rope":
            Q = self.rotary(Q)
            K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, 
                 dropout: float = 0.1, norm_type: str = "rmsnorm", pos_encoding: str = "rope"):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, pos_encoding)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Choose normalization type
        if norm_type == "rmsnorm":
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
        else:  # layernorm
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Add learned positional embeddings if specified
        if config.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, 
                           config.max_seq_len, config.dropout, 
                           config.norm_type, config.pos_encoding)
            for _ in range(config.n_layers)
        ])

        # Choose final normalization
        if config.norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(config.d_model)
        else:
            self.norm = nn.LayerNorm(config.d_model)
            
        self.output_dropout = nn.Dropout(config.dropout)

        # Output layer with optional weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.use_weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        
        # Add learned positional embeddings if specified
        if self.config.pos_encoding == "learned":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.position_embedding(positions)
            
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {
        'val_loss': avg_loss, 
        'val_accuracy': accuracy, 
        'val_perplexity': perplexity,
        'total_tokens': total_tokens
    }

def setup_optimizer(model: nn.Module, config: ModelConfig):
    """Setup optimizer based on configuration"""
    if config.optimizer_type == "muon_hybrid":
        return setup_muon_optimizer(model, config)
    elif config.optimizer_type == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=config.adamw_lr, weight_decay=config.weight_decay)]
    elif config.optimizer_type == "sgd":
        return [torch.optim.SGD(model.parameters(), lr=config.sgd_lr, momentum=config.sgd_momentum, weight_decay=config.weight_decay)]
    elif config.optimizer_type == "adam":
        return [torch.optim.Adam(model.parameters(), lr=config.adamw_lr, weight_decay=config.weight_decay)]
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95, ns_steps=config.ns_steps)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def setup_lr_scheduler(optimizer, config: ModelConfig):
    """Setup learning rate scheduler based on configuration"""
    if config.lr_schedule == "cosine_warmup":
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif config.lr_schedule == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config.max_steps)
    
    elif config.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.max_steps//3, gamma=0.5)
    
    else:  # constant
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader, 
                experiment_name: str = "default"):
    """Enhanced training function with detailed logging"""
    print(f"\n🚀 Training model: {experiment_name}")
    print(f"   Optimizer: {config.optimizer_type}")
    print(f"   Normalization: {config.norm_type}")
    print(f"   Positional Encoding: {config.pos_encoding}")
    print(f"   Weight Tying: {config.use_weight_tying}")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers and schedulers
    optimizers = setup_optimizer(model, config)
    schedulers = [setup_lr_scheduler(opt, config) for opt in optimizers]

    scaler = GradScaler() if config.use_amp else None

    # Training metrics tracking
    metrics_history = {
        'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_perplexity': [],
        'learning_rates': [], 'grad_norms': [], 'step_times': []
    }

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc=f"Training {experiment_name}")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            step_start_time = time.time()
            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                # Record metrics
                metrics_history['grad_norms'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                metrics_history['learning_rates'].append(optimizers[0].param_groups[0]['lr'])

            step_time = time.time() - step_start_time
            metrics_history['step_times'].append(step_time)

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))
                    
                    metrics_history['train_loss'].append(current_loss)

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}',
                    'step_time': f'{step_time:.3f}s'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                metrics_history['val_loss'].append(eval_metrics['val_loss'])
                metrics_history['val_accuracy'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexity'].append(eval_metrics['val_perplexity'])

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    # Calculate training statistics
    avg_step_time = sum(metrics_history['step_times']) / len(metrics_history['step_times'])
    tokens_per_sec = (config.batch_size * config.max_seq_len) / avg_step_time
    
    final_eval.update({
        'training_time': training_time,
        'avg_step_time': avg_step_time,
        'tokens_per_sec': tokens_per_sec,
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'metrics_history': metrics_history
    })

    return model, final_eval

class ExperimentRunner:
    """Class to run and manage experiments"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.results = defaultdict(list)
        
        # Create output directory
        self.output_dir = Path(experiment_config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_config.experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"📁 Experiment directory: {self.experiment_dir}")

    def run_optimizer_comparison(self):
        """Run optimizer comparison experiments"""
        print("\n🔬 Running Optimizer Comparison Experiments")
        
        optimizers = ["muon_hybrid", "adamw", "sgd", "adam"]
        newton_schulz_steps = [1, 3, 5, 7, 10]
        learning_rates = {
            "muon_hybrid": [0.005, 0.01, 0.02, 0.03],
            "adamw": [0.0005, 0.001, 0.002, 0.003],
            "sgd": [0.005, 0.01, 0.02, 0.03],
            "adam": [0.0005, 0.001, 0.002, 0.003]
        }
        
        base_config = ModelConfig.get_model_size_config("small")
        base_config.max_steps = 1000  # Shorter runs for comparison
        
        # Load data once
        texts, tokenizer, tokens = load_and_cache_data(base_config)
        dataset = TextTokenDataset(tokens, base_config.max_seq_len)
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        # 1. Basic optimizer comparison
        for optimizer in optimizers:
            for seed in self.config.seeds[:2]:  # Fewer seeds for speed
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.optimizer_type = optimizer
                
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"optimizer_{optimizer}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results[f"optimizer_comparison"].append({
                    'optimizer': optimizer,
                    'seed': seed,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'tokens_per_sec': results['tokens_per_sec'],
                    'best_val_loss': results['best_val_loss']
                })
                
                if not self.config.save_models:
                    del model  # Free memory
        
        # 2. Newton-Schulz steps ablation (Muon only)
        print("\n🔬 Newton-Schulz Steps Ablation")
        for ns_steps in newton_schulz_steps:
            for seed in self.config.seeds[:2]:
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.optimizer_type = "muon_hybrid"
                config.ns_steps = ns_steps
                
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"ns_steps_{ns_steps}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results[f"ns_steps_ablation"].append({
                    'ns_steps': ns_steps,
                    'seed': seed,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'best_val_loss': results['best_val_loss']
                })
                
                del model
        
        # 3. Learning rate sensitivity
        print("\n🔬 Learning Rate Sensitivity")
        for optimizer in ["muon_hybrid", "adamw"]:
            for lr in learning_rates[optimizer]:
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.optimizer_type = optimizer
                if optimizer == "muon_hybrid":
                    config.muon_lr = lr
                else:
                    config.adamw_lr = lr
                
                set_seed(42)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"lr_sensitivity_{optimizer}_lr{lr}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results[f"lr_sensitivity"].append({
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'best_val_loss': results['best_val_loss'],
                    'training_stable': results['val_loss'] < 10.0  # Basic stability check
                })
                
                del model

    def run_architecture_ablation(self):
        """Run architecture ablation studies"""
        print("\n🔬 Running Architecture Ablation Studies")
        
        base_config = ModelConfig.get_model_size_config("small")
        base_config.max_steps = 1000
        
        # Load data
        texts, tokenizer, tokens = load_and_cache_data(base_config)
        dataset = TextTokenDataset(tokens, base_config.max_seq_len)
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        # 1. Normalization comparison
        print("🔬 Normalization Comparison")
        for norm_type in ["rmsnorm", "layernorm"]:
            for seed in self.config.seeds[:2]:
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.norm_type = norm_type
                
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"norm_{norm_type}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results["normalization_comparison"].append({
                    'norm_type': norm_type,
                    'seed': seed,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'best_val_loss': results['best_val_loss']
                })
                
                del model
        
        # 2. Positional encoding comparison
        print("🔬 Positional Encoding Comparison")
        for pos_encoding in ["rope", "sinusoidal", "learned", "none"]:
            for seed in self.config.seeds[:2]:
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.pos_encoding = pos_encoding
                
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"pos_enc_{pos_encoding}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results["pos_encoding_comparison"].append({
                    'pos_encoding': pos_encoding,
                    'seed': seed,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'best_val_loss': results['best_val_loss']
                })
                
                del model
        
        # 3. Weight tying impact
        print("🔬 Weight Tying Impact")
        for use_weight_tying in [True, False]:
            for seed in self.config.seeds[:2]:
                config = ModelConfig.get_model_size_config("small")
                config.max_steps = 1000
                config.use_weight_tying = use_weight_tying
                
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"weight_tying_{use_weight_tying}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results["weight_tying_impact"].append({
                    'use_weight_tying': use_weight_tying,
                    'seed': seed,
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'total_params': results['total_params'],
                    'best_val_loss': results['best_val_loss']
                })
                
                del model

    def run_scaling_study(self):
        """Run scaling studies"""
        print("\n🔬 Running Scaling Studies")
        
        # Model size scaling
        model_sizes = ["tiny", "small", "medium"]
        
        for size in model_sizes:
            config = ModelConfig.get_model_size_config(size)
            config.max_steps = 1000
            
            # Adjust data size for larger models
            if size == "medium":
                config.max_tokens = 750000
                config.num_documents = 3000
            
            texts, tokenizer, tokens = load_and_cache_data(config)
            dataset = TextTokenDataset(tokens, config.max_seq_len)
            val_size = len(dataset) // 10
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            
            for seed in self.config.seeds[:2]:
                set_seed(seed)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
                
                experiment_name = f"scaling_{size}_seed{seed}"
                model, results = train_model(config, train_loader, val_loader, experiment_name)
                
                self.results["model_scaling"].append({
                    'model_size': size,
                    'seed': seed,
                    'total_params': results['total_params'],
                    'final_val_loss': results['val_loss'],
                    'final_val_accuracy': results['val_accuracy'],
                    'training_time': results['training_time'],
                    'tokens_per_sec': results['tokens_per_sec'],
                    'best_val_loss': results['best_val_loss']
                })
                
                del model

    def run_efficiency_experiments(self):
        """Run computational efficiency experiments"""
        print("\n🔬 Running Efficiency Experiments")
        
        base_config = ModelConfig.get_model_size_config("small")
        base_config.max_steps = 500  # Shorter for efficiency tests
        
        texts, tokenizer, tokens = load_and_cache_data(base_config)
        dataset = TextTokenDataset(tokens, base_config.max_seq_len)
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        # Mixed precision comparison
        for use_amp in [True, False]:
            config = ModelConfig.get_model_size_config("small")
            config.max_steps = 500
            config.use_amp = use_amp
            
            set_seed(42)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
            
            experiment_name = f"amp_{use_amp}"
            model, results = train_model(config, train_loader, val_loader, experiment_name)
            
            self.results["mixed_precision"].append({
                'use_amp': use_amp,
                'final_val_loss': results['val_loss'],
                'training_time': results['training_time'],
                'tokens_per_sec': results['tokens_per_sec'],
                'avg_step_time': results['avg_step_time']
            })
            
            del model

    def save_results(self):
        """Save experimental results"""
        results_file = self.experiment_dir / "results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for experiment_type, experiments in self.results.items():
            json_results[experiment_type] = experiments
            
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"💾 Results saved to {results_file}")
        
        # Save summary
        self.create_summary()

    def create_summary(self):
        """Create experiment summary"""
        summary_file = self.experiment_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Experiment Summary: {self.config.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for experiment_type, experiments in self.results.items():
                f.write(f"{experiment_type.upper()}\n")
                f.write("-" * 30 + "\n")
                
                if experiments:
                    # Calculate averages for numeric fields
                    numeric_fields = ['final_val_loss', 'final_val_accuracy', 'training_time', 'tokens_per_sec']
                    
                    for field in numeric_fields:
                        if field in experiments[0]:
                            values = [exp[field] for exp in experiments if field in exp]
                            if values:
                                avg_val = sum(values) / len(values)
                                f.write(f"Average {field}: {avg_val:.4f}\n")
                    
                    f.write(f"Number of experiments: {len(experiments)}\n")
                f.write("\n")

    def plot_results(self):
        """Create plots of experimental results"""
        if not self.config.plot_results:
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Plot optimizer comparison
            if "optimizer_comparison" in self.results:
                self._plot_optimizer_comparison()
            
            # Plot architecture ablation
            if "normalization_comparison" in self.results:
                self._plot_architecture_results()
                
            # Plot scaling results
            if "model_scaling" in self.results:
                self._plot_scaling_results()
                
            print(f"📊 Plots saved to {self.experiment_dir}")
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available for plotting")

    def _plot_optimizer_comparison(self):
        """Plot optimizer comparison results"""
        data = self.results["optimizer_comparison"]
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Optimizer Comparison Results')
        
        # Validation loss
        sns.boxplot(data=df, x='optimizer', y='final_val_loss', ax=axes[0,0])
        axes[0,0].set_title('Final Validation Loss')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Validation accuracy
        sns.boxplot(data=df, x='optimizer', y='final_val_accuracy', ax=axes[0,1])
        axes[0,1].set_title('Final Validation Accuracy')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Training time
        sns.boxplot(data=df, x='optimizer', y='training_time', ax=axes[1,0])
        axes[1,0].set_title('Training Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Throughput
        sns.boxplot(data=df, x='optimizer', y='tokens_per_sec', ax=axes[1,1])
        axes[1,1].set_title('Training Throughput (tokens/sec)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "optimizer_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_architecture_results(self):
        """Plot architecture ablation results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Architecture Ablation Results')
        
        # Normalization comparison
        if "normalization_comparison" in self.results:
            norm_data = pd.DataFrame(self.results["normalization_comparison"])
            sns.boxplot(data=norm_data, x='norm_type', y='final_val_loss', ax=axes[0])
            axes[0].set_title('Normalization: Validation Loss')
        
        # Positional encoding comparison
        if "pos_encoding_comparison" in self.results:
            pos_data = pd.DataFrame(self.results["pos_encoding_comparison"])
            sns.boxplot(data=pos_data, x='pos_encoding', y='final_val_loss', ax=axes[1])
            axes[1].set_title('Positional Encoding: Validation Loss')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Weight tying impact
        if "weight_tying_impact" in self.results:
            wt_data = pd.DataFrame(self.results["weight_tying_impact"])
            sns.boxplot(data=wt_data, x='use_weight_tying', y='final_val_loss', ax=axes[2])
            axes[2].set_title('Weight Tying: Validation Loss')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "architecture_ablation.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scaling_results(self):
        """Plot scaling study results"""
        data = pd.DataFrame(self.results["model_scaling"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Scaling Results')
        
        # Parameters vs Performance
        sns.scatterplot(data=data, x='total_params', y='final_val_loss', 
                       hue='model_size', size='model_size', ax=axes[0,0])
        axes[0,0].set_title('Parameters vs Validation Loss')
        axes[0,0].set_xscale('log')
        
        # Parameters vs Training Time
        sns.scatterplot(data=data, x='total_params', y='training_time', 
                       hue='model_size', size='model_size', ax=axes[0,1])
        axes[0,1].set_title('Parameters vs Training Time')
        axes[0,1].set_xscale('log')
        
        # Model size comparison - Loss
        sns.boxplot(data=data, x='model_size', y='final_val_loss', ax=axes[1,0])
        axes[1,0].set_title('Model Size vs Validation Loss')
        
        # Model size comparison - Throughput
        sns.boxplot(data=data, x='model_size', y='tokens_per_sec', ax=axes[1,1])
        axes[1,1].set_title('Model Size vs Throughput')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "scaling_results.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_all_experiments(self):
        """Run all configured experiments"""
        print(f"🚀 Starting comprehensive experiments: {self.config.experiment_name}")
        
        if self.config.experiment_type == "optimizer_comparison" or self.config.experiment_type == "all":
            self.run_optimizer_comparison()
            
        if self.config.experiment_type == "architecture_ablation" or self.config.experiment_type == "all":
            self.run_architecture_ablation()
            
        if self.config.experiment_type == "scaling_study" or self.config.experiment_type == "all":
            self.run_scaling_study()
            
        if self.config.experiment_type == "efficiency" or self.config.experiment_type == "all":
            self.run_efficiency_experiments()
        
        # Save and visualize results
        self.save_results()
        self.plot_results()
        
        print(f"\n🎉 All experiments completed!")
        print(f"📁 Results saved in: {self.experiment_dir}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='LLM Training Experiments')
    parser.add_argument('--experiment-type', type=str, default='single',
                       choices=['single', 'optimizer_comparison', 'architecture_ablation', 
                               'scaling_study', 'efficiency', 'all'],
                       help='Type of experiment to run')
    parser.add_argument('--experiment-name', type=str, default='llm_experiments',
                       help='Name for the experiment')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per experiment')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.experiment_type == 'single':
        # Run single experiment (original behavior)
        set_seed(42)
        
        config = ModelConfig()
        print(f"\n📋 Model Configuration:")
        print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
        print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
        print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

        # Load data
        texts, tokenizer, tokens = load_and_cache_data(config)
        dataset = TextTokenDataset(tokens, config.max_seq_len)

        # Train/val split
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

        # Train model
        start_time = time.time()
        model, final_metrics = train_model(config, train_loader, val_loader, "single_run")
        total_time = time.time() - start_time

        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
        print(f"   Throughput: {final_metrics['tokens_per_sec']:.0f} tokens/sec")
        
    else:
        # Run comprehensive experiments
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            experiment_type=args.experiment_type,
            output_dir=args.output_dir,
            num_runs=args.num_runs,
            save_models=args.save_models,
            plot_results=not args.no_plots
        )
        
        runner = ExperimentRunner(experiment_config)
        runner.run_all_experiments()

if __name__ == "__main__":
    main()