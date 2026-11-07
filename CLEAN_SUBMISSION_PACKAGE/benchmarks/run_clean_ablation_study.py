"""
Clean, Rigorous Multi-Seed Ablation Study for IVERI-HYPERION
============================================================
Evaluates 4 variations across 3 independent random seeds (42, 43, 44):
1. Full IVERI-HYPERION (Baseline, memory_dim=64, full entropy gating)
2. Ablated Titans Memory (Titans module bypassed: x = x + 0.0)
3. Reduced Memory Dimension (memory_dim=32)
4. Ablated Entropy Gating (Uniform compute: gates clamped to 1.0)

Academic Controls:
- Identical TinyStories English dataset (processed val.pt / train.pt)
- Exact same batch sequences per step per seed
- Fixed 200 lockstep optimization steps
- Bug-fixed eval_model with exact token counting (no 4% batch overflow bug)
- Saves full results to reports/clean_titans_ablation_results.json
"""

import sys
import os
import math
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "CLEAN_SUBMISSION_PACKAGE"))
sys.path.insert(0, str(REPO_ROOT))

from models.iveri_v1 import IVERIv1Model, IVERIv1Block, BYTE_VOCAB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device: {device}]", flush=True)

# ── Load Dataset ──────────────────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data" / "processed"
train_pt = torch.load(DATA_DIR / "train.pt", map_location="cpu", weights_only=False)
val_pt   = torch.load(DATA_DIR / "val.pt",   map_location="cpu", weights_only=False)

train_inputs  = train_pt["inputs"]
train_targets = train_pt["targets"]
val_inputs    = val_pt["inputs"]
val_targets   = val_pt["targets"]

BATCH_SIZE  = 4
SEQ_LEN     = 512
TOTAL_STEPS = 200
LR          = 1e-3
SEEDS       = [42, 43, 44]

# ── Ablation Block Wrappers ───────────────────────────────────────────────────
class AblatedTitansBlock(IVERIv1Block):
    """Bypasses Titans associative memory entirely."""
    def forward(self, x, entropy):
        gate = torch.sigmoid(self.gate_proj(entropy))
        x = x + gate * self.mamba(self.norm1(x))
        # Titans bypassed: x = x + 0.0
        ffn_gate = torch.sigmoid(self.ffn_gate_proj(entropy))
        x = x + ffn_gate * self.ffn(self.norm3(x))
        return x

class AblatedEntropyBlock(IVERIv1Block):
    """Disables entropy gating; enforces uniform 1.0 residual compute."""
    def forward(self, x, entropy):
        # Gates clamped to 1.0
        x = x + 1.0 * self.mamba(self.norm1(x))
        x = x + self.titans(self.norm2(x), entropy)
        x = x + 1.0 * self.ffn(self.norm3(x))
        return x

def build_model(variant_name):
    if variant_name == "Full IVERI-HYPERION (Baseline)":
        return IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64)
    elif variant_name == "Ablated Titans Memory (No-Titans)":
        m = IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64)
        for i in range(len(m.blocks)):
            ab_block = AblatedTitansBlock(hidden_dim=406, memory_dim=64)
            ab_block.load_state_dict(m.blocks[i].state_dict())
            m.blocks[i] = ab_block
        return m
    elif variant_name == "Reduced Memory Dimension (d_mem=32)":
        return IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=32)
    elif variant_name == "Ablated Entropy Gating (Uniform Compute)":
        m = IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64)
        for i in range(len(m.blocks)):
            ab_block = AblatedEntropyBlock(hidden_dim=406, memory_dim=64)
            ab_block.load_state_dict(m.blocks[i].state_dict())
            m.blocks[i] = ab_block
        return m
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

# ── Bug-Fixed Exact Evaluation Helper ─────────────────────────────────────────
def eval_model(model, max_seqs=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = min(max_seqs, val_inputs.shape[0])
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch_slice = slice(i, min(i + BATCH_SIZE, n))
            v_in = val_inputs[batch_slice].to(device)
            v_tg = val_targets[batch_slice].to(device)
            bsz, slen = v_in.shape
            out = model(v_in, return_dict=False)
            loss_sum = F.cross_entropy(out.reshape(-1, BYTE_VOCAB), v_tg.reshape(-1), reduction="sum").item()
            total_loss += loss_sum
            total_tokens += bsz * slen
    return total_loss / total_tokens

# ── Train Single Model ────────────────────────────────────────────────────────
def train_variant(variant_name, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(variant_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
    
    def lr_lambda(step):
        warmup = 10
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        prog = float(step + 1 - warmup) / float(max(1, TOTAL_STEPS - warmup))
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * prog)))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Use fixed seed for training batch order so all variants see exact same sequences
    batch_gen = torch.Generator().manual_seed(seed * 1000 + 7)

    model.train()
    for step in range(1, TOTAL_STEPS + 1):
        idx = torch.randint(0, train_inputs.shape[0], (BATCH_SIZE,), generator=batch_gen)
        b_in = train_inputs[idx].to(device)
        b_tg = train_targets[idx].to(device)

        opt.zero_grad()
        out = model(b_in, return_dict=False)
        loss = F.cross_entropy(out.reshape(-1, BYTE_VOCAB), b_tg.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    val_loss = eval_model(model)
    byte_ppl = math.exp(val_loss)
    return val_loss, byte_ppl

# ── Main Campaign Execution ───────────────────────────────────────────────────
VARIANTS = [
    "Full IVERI-HYPERION (Baseline)",
    "Ablated Titans Memory (No-Titans)",
    "Reduced Memory Dimension (d_mem=32)",
    "Ablated Entropy Gating (Uniform Compute)"
]

print("=" * 80)
print("  STARTING CLEAN MULTI-SEED ABLATION STUDY (IVERI-HYPERION 10M)")
print("=" * 80)

results = {}

for variant in VARIANTS:
    print(f"\nEvaluating Variant: {variant}")
    seed_losses = []
    seed_ppls = []

    for seed in SEEDS:
        t0 = time.time()
        v_loss, v_ppl = train_variant(variant, seed)
        dt = time.time() - t0
        seed_losses.append(v_loss)
        seed_ppls.append(v_ppl)
        print(f"  Seed {seed}: Val Loss = {v_loss:.4f} | Byte-PPL = {v_ppl:.2f} ({dt:.1f}s)")

    mean_loss = sum(seed_losses) / len(seed_losses)
    std_loss  = (sum((x - mean_loss)**2 for x in seed_losses) / (len(seed_losses) - 1)) ** 0.5
    mean_ppl  = sum(seed_ppls) / len(seed_ppls)
    std_ppl   = (sum((x - mean_ppl)**2 for x in seed_ppls) / (len(seed_ppls) - 1)) ** 0.5

    results[variant] = {
        "seeds": {str(s): {"val_loss": l, "byte_ppl": p} for s, l, p in zip(SEEDS, seed_losses, seed_ppls)},
        "loss_mean": round(mean_loss, 4),
        "loss_std":  round(std_loss, 4),
        "ppl_mean":  round(mean_ppl, 2),
        "ppl_std":   round(std_ppl, 2)
    }
    print(f"  --> Mean Loss: {mean_loss:.4f} +/- {std_loss:.4f} | Mean Byte-PPL: {mean_ppl:.2f} +/- {std_ppl:.2f}")

# Compute relative changes against baseline
base_loss = results["Full IVERI-HYPERION (Baseline)"]["loss_mean"]
for variant in VARIANTS:
    v_loss = results[variant]["loss_mean"]
    # Relative percentage degradation compared to baseline
    rel_pct = ((v_loss - base_loss) / base_loss) * 100.0
    results[variant]["rel_delta_pct_vs_baseline"] = round(rel_pct, 2)

OUTPUT_FILE = REPO_ROOT / "CLEAN_SUBMISSION_PACKAGE" / "reports" / "clean_titans_ablation_results.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"[SUCCESS] Clean Ablation Results saved to: {OUTPUT_FILE}")
print("=" * 80)
