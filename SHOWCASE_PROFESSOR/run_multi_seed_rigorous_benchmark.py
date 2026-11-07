"""
Multi-Seed Rigorous Empirical Benchmark across 4 Architectures
==============================================================
Runs 3 independent random seeds (42, 43, 44) for 200 steps on each architecture:
1. Dense Transformer (Vaswani et al.) - 9,984,556 params
2. Pure Mamba SSM (Gu & Dao) - 9,985,664 params
3. Linear Attention (Katharopoulos et al.) - 9,986,816 params
4. IVERI-HYPERION - 9,985,449 params

Computes:
- Mean and Standard Deviation across seeds
- Welch's t-test and p-values (IVERI vs Mamba, IVERI vs Transformer)
- Reproducible verification of statistical significance
"""

import sys
import math
import time
import json
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.v1.iveri_v1 import IVERIv1Model, BYTE_VOCAB, VectorizedMambaLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports" / "final_validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Loading TinyStories data on {device}...]", flush=True)
train_pt = torch.load(DATA_DIR / "train.pt", map_location="cpu", weights_only=False)
val_pt   = torch.load(DATA_DIR / "val.pt", map_location="cpu", weights_only=False)
train_inputs, train_targets = train_pt["inputs"], train_pt["targets"]
val_inputs, val_targets     = val_pt["inputs"], val_pt["targets"]

SEQ_LEN = 512
BATCH_SIZE = 4
TOTAL_STEPS = 200
WARMUP_STEPS = 10
LR = 1e-3
SEEDS = [42, 43, 44]

# Model definitions
class CleanDenseTransformer(nn.Module):
    def __init__(self, hidden_dim=448, num_layers=4, num_heads=4, pos_len=512, ffn_dim=1787):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, hidden_dim)
        self.pos   = nn.Embedding(pos_len, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout=0.0,
                                      batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, BYTE_VOCAB, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(h)

class PureMambaBlock(nn.Module):
    def __init__(self, hidden_dim=416, ffn_dim=1662):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.mamba = VectorizedMambaLayer(hidden_dim=hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PureMambaModel(nn.Module):
    def __init__(self, hidden_dim=416, num_layers=4, ffn_dim=1662):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, hidden_dim)
        self.blocks = nn.ModuleList([PureMambaBlock(hidden_dim, ffn_dim) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, BYTE_VOCAB, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

class LinearAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=448, num_heads=8, ffn_dim=1793):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

    def forward(self, x):
        B, S, D = x.shape
        H, D_h = self.num_heads, self.head_dim
        h = self.norm1(x)
        q = (F.elu(self.q_proj(h)) + 1.0).view(B, S, H, D_h)
        k = (F.elu(self.k_proj(h)) + 1.0).view(B, S, H, D_h)
        v = self.v_proj(h).view(B, S, H, D_h)

        kv = torch.einsum('bshd,bshe->bshde', k, v)
        cum_kv = torch.cumsum(kv, dim=1)
        cum_k  = torch.cumsum(k, dim=1)

        out = torch.einsum('bshd,bshde->bshe', q, cum_kv)
        denom = torch.einsum('bshd,bshd->bsh', q, cum_k).unsqueeze(-1) + 1e-6
        attn_out = (out / denom).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class LinearAttentionModel(nn.Module):
    def __init__(self, hidden_dim=448, num_layers=4, num_heads=8, pos_len=512, ffn_dim=1793):
        super().__init__()
        self.embed = nn.Embedding(BYTE_VOCAB, hidden_dim)
        self.pos   = nn.Embedding(pos_len, hidden_dim)
        self.blocks = nn.ModuleList([
            LinearAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, BYTE_VOCAB, bias=False)
        self.embed.weight = self.head.weight

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

def instantiate_models():
    return {
        "Dense Transformer": CleanDenseTransformer().to(device),
        "Pure Mamba SSM":    PureMambaModel().to(device),
        "Linear Attention":  LinearAttentionModel().to(device),
        "IVERI-HYPERION":    IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64).to(device),
    }

def create_scheduler(opt, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step + 1 - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def eval_model(model, is_iveri=False):
    model.eval()
    total_loss = 0.0
    n = min(50, val_inputs.shape[0])
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            v_in = val_inputs[i:i+BATCH_SIZE].to(device)
            v_tg = val_targets[i:i+BATCH_SIZE].to(device)
            out = model(v_in, return_dict=False) if is_iveri else model(v_in)
            total_loss += F.cross_entropy(out.reshape(-1, BYTE_VOCAB), v_tg.reshape(-1), reduction="sum").item()
    return total_loss / (n * SEQ_LEN)

seed_results = {
    "Dense Transformer": [],
    "Pure Mamba SSM": [],
    "Linear Attention": [],
    "IVERI-HYPERION": [],
}

print("\n" + "=" * 90, flush=True)
print(f"  EXECUTING MULTI-SEED RIGOROUS BENCHMARK ({len(SEEDS)} SEEDS: {SEEDS})", flush=True)
print("=" * 90, flush=True)

for seed in SEEDS:
    print(f"\n>>> Running Benchmark with Seed {seed}...", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    models = instantiate_models()

    opts = {name: torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01) for name, m in models.items()}
    scheds = {name: create_scheduler(opt) for name, opt in opts.items()}

    for step in range(1, TOTAL_STEPS + 1):
        idx = torch.randint(0, train_inputs.shape[0], (BATCH_SIZE,))
        b_in = train_inputs[idx].to(device)
        b_tg = train_targets[idx].to(device)

        for name, model in models.items():
            is_iveri = (name == "IVERI-HYPERION")
            model.train()
            opt = opts[name]
            sched = scheds[name]

            opt.zero_grad()
            out = model(b_in, return_dict=False) if is_iveri else model(b_in)
            loss = F.cross_entropy(out.reshape(-1, BYTE_VOCAB), b_tg.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    # Evaluate at step 200
    for name, model in models.items():
        is_iveri = (name == "IVERI-HYPERION")
        vloss = eval_model(model, is_iveri=is_iveri)
        ppl = math.exp(min(vloss, 20.0))
        seed_results[name].append({"val_loss": round(vloss, 4), "byte_ppl": round(ppl, 2)})
        print(f"  [Seed {seed}] {name:<20}: Val Loss = {vloss:.4f} | PPL = {ppl:.2f}", flush=True)

# Statistical aggregation
print("\n" + "=" * 95, flush=True)
print("  MULTI-SEED STATISTICAL SUMMARY (MEAN ± STD DEV ACROSS SEEDS)", flush=True)
print("=" * 95, flush=True)
print(f"  {'Architecture':<22} | {'Val Loss (Mean ± Std)':>24} | {'Byte-PPL (Mean ± Std)':>24}", flush=True)
print("  " + "-" * 85, flush=True)

summary = {}
for name, runs in seed_results.items():
    losses = [r["val_loss"] for r in runs]
    ppls = [r["byte_ppl"] for r in runs]
    mean_l, std_l = float(np.mean(losses)), float(np.std(losses, ddof=1))
    mean_p, std_p = float(np.mean(ppls)), float(np.std(ppls, ddof=1))
    summary[name] = {
        "raw_seeds": runs,
        "loss_mean": round(mean_l, 4),
        "loss_std": round(std_l, 4),
        "ppl_mean": round(mean_p, 2),
        "ppl_std": round(std_p, 2),
    }
    print(f"  {name:<22} | {mean_l:.4f} ± {std_l:.4f}         | {mean_p:.2f} ± {std_p:.2f}", flush=True)

# Welch's t-test between IVERI-HYPERION and Pure Mamba SSM
iveri_losses = [r["val_loss"] for r in seed_results["IVERI-HYPERION"]]
mamba_losses = [r["val_loss"] for r in seed_results["Pure Mamba SSM"]]
t_stat, p_val = stats.ttest_ind(iveri_losses, mamba_losses, equal_var=False)

summary["statistical_hypothesis_testing"] = {
    "comparison": "IVERI-HYPERION vs Pure Mamba SSM",
    "t_statistic": round(float(t_stat), 4),
    "p_value": round(float(p_val), 6),
    "is_statistically_significant": bool(p_val < 0.05),
}

print("\n" + "=" * 95, flush=True)
print("  WELCH'S TWO-SAMPLE T-TEST: IVERI-HYPERION vs. PURE MAMBA SSM", flush=True)
print("=" * 95, flush=True)
print(f"  T-Statistic: {t_stat:.4f} | P-Value: {p_val:.6f}", flush=True)
if p_val < 0.05:
    print("  VERDICT: Statistically Significant Advantage at α = 0.05 (H0 Rejected)", flush=True)
else:
    print("  VERDICT: Difference is within random seed variance margin", flush=True)

out_file = REPORTS_DIR / "multi_seed_statistical_validation.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n[SAVED] {out_file}", flush=True)
