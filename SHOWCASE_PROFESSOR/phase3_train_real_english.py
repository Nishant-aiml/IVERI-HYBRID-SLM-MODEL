#!/usr/bin/env python3
"""
========================================================================================
IVERI-HYPERION: PHASE 3 — REAL ENGLISH TRAINING & CAPABILITY BENCHMARK
========================================================================================
Trains the unified IVERI-HYPERION architecture (Mamba SSM + Zero-Init Titans Memory +
Vectorized Byte Entropy Router) on real English text (TinyStories).

Key Features:
  - Strict 10.0M Parameter Match (9,985,441 params)
  - Zero-initialized Titans memory projection (stabilizes early language learning)
  - Real English next-byte Cross-Entropy loss & Perplexity tracking
  - Live text generation sample logged at every 100-step checkpoint
  - Saves trained checkpoint: checkpoints/iveri_hyperion_real_english.pt

Outputs Generated:
  - reports/capability/iveri_hyperion_real_english_results.json
  - checkpoints/iveri_hyperion_real_english.pt
  - SHOWCASE_PROFESSOR/paper1_real_english_loss_curves.png
========================================================================================
"""

import sys, os, time, math, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Directory resolution
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from model.v1.iveri_v1 import IVERIv1Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 85)
print("  IVERI-HYPERION: PHASE 3 — REAL ENGLISH TRAINING & BENCHMARK")
print(f"  Target Architecture: IVERI-HYPERION (ENSN - Entropy-Driven Neural State Network)")
print(f"  Execution Device: {device} | PyTorch: {torch.__version__}")
print("=" * 85)

BYTE_VOCAB = 259
CHECKPOINT_DIR = root_dir / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
REPORTS_DIR = root_dir / "reports" / "capability"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Real English Data Tensors
DATA_DIR = root_dir / "data" / "processed"
train_data_path = DATA_DIR / "train.pt"
val_data_path = DATA_DIR / "val.pt"

if not train_data_path.exists() or not val_data_path.exists():
    raise FileNotFoundError(f"Processed dataset not found. Run phase2_download_dataset.py first!")

print(f"\n[1/4] Loading Real English Datasets...")
train_pt = torch.load(train_data_path, map_location="cpu")
val_pt = torch.load(val_data_path, map_location="cpu")

train_inputs = train_pt["inputs"]   # (N_train, 512)
train_targets = train_pt["targets"] # (N_train, 512)
val_inputs = val_pt["inputs"]       # (N_val, 512)
val_targets = val_pt["targets"]     # (N_val, 512)

print(f"  --> Train Set: {train_inputs.shape[0]:,} sequences ({train_inputs.shape[0] * 512:,} tokens)")
print(f"  --> Val Set:   {val_inputs.shape[0]:,} sequences ({val_inputs.shape[0] * 512:,} tokens)")

# 2. Build Model
print(f"\n[2/4] Initializing IVERI-HYPERION Model (10M Parameters)...")
model = IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64).to(device)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  --> Trainable Parameters: {param_count:,} (Strict 10M Budget: {((param_count-1e7)/1e7)*100:+.2f}%)")

# Verify zero-init on Titans
out_weight_norm = model.blocks[0].titans.out_proj.weight.abs().sum().item()
print(f"  --> Block 0 Titans out_proj sum: {out_weight_norm:.6f} (Must be 0.0 for stable start)")

# 3. Setup Optimizer & Mixed Precision
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
TOTAL_STEPS = 1000
ACCUM_STEPS = 4  # Effective batch size = 4 * 4 = 16 sequences = 8,192 tokens/step
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
BATCH_SIZE = 4
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

def generate_sample(prompt_str, max_len=50, temp=0.75, top_p=0.90, rep_pen=1.25):
    model.eval()
    prompt_bytes = list(prompt_str.encode("utf-8"))
    x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    gen = []
    for _ in range(max_len):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(x, return_dict=False)[:, -1, :256].squeeze(0).float()
            if gen:
                for past_b in set(gen[-20:]):
                    if logits[past_b] > 0:
                        logits[past_b] /= rep_pen
                    else:
                        logits[past_b] *= rep_pen
            scaled = logits / max(temp, 1e-4)
            sorted_l, sorted_i = torch.sort(scaled, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scaled[sorted_i[remove]] = -1e4
            probs = F.softmax(scaled, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = torch.ones_like(probs) / probs.numel()
            else:
                probs = probs / probs.sum()
            nb = torch.multinomial(probs, 1).item()
        x = torch.cat([x, torch.tensor([[nb]], dtype=torch.long, device=device)], dim=1)
        gen.append(nb)
    model.train()
    return bytes(x[0].tolist()).decode("utf-8", errors="replace")

# 4. Training Loop with AMP and Gradient Accumulation
print(f"\n[3/4] Starting 1000-Step Accelerated Training on Real English...")
print(f"  Micro-Batch: {BATCH_SIZE} | Accum Steps: {ACCUM_STEPS} | Effective Batch: {BATCH_SIZE * ACCUM_STEPS} (8,192 tokens/step)")
print(f"  Mixed Precision: FP16 Tensor Core Accelerated | Optimizer: AdamW + CosineAnnealing")
print("-" * 85)

history = []
t_start = time.perf_counter()
tokens_processed = 0

model.train()
optimizer.zero_grad()

for step in range(1, TOTAL_STEPS + 1):
    step_loss = 0.0
    for _ in range(ACCUM_STEPS):
        idx = torch.randint(0, train_inputs.shape[0], (BATCH_SIZE,))
        b_in = train_inputs[idx].to(device)
        b_tg = train_targets[idx].to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(b_in, return_dict=False)
            loss = F.cross_entropy(logits.reshape(-1, BYTE_VOCAB), b_tg.reshape(-1)) / ACCUM_STEPS
            
        scaler.scale(loss).backward()
        step_loss += loss.item() * ACCUM_STEPS
        tokens_processed += BATCH_SIZE * 512
        
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()
    
    # Checkpoint evaluation every 100 steps
    if step % 100 == 0 or step == TOTAL_STEPS:
        model.eval()
        val_loss = 0.0
        val_samples = min(25, val_inputs.shape[0])
        with torch.no_grad():
            for v_idx in range(0, val_samples, BATCH_SIZE):
                v_in = val_inputs[v_idx:v_idx+BATCH_SIZE].to(device)
                v_tg = val_targets[v_idx:v_idx+BATCH_SIZE].to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    v_out = model(v_in, return_dict=False)
                    val_loss += F.cross_entropy(v_out.reshape(-1, BYTE_VOCAB), v_tg.reshape(-1), reduction="sum").item()
        val_loss /= (val_samples * 512)
        val_ppl = math.exp(min(val_loss, 20.0))
        elapsed = time.perf_counter() - t_start
        tok_rate = tokens_processed / max(elapsed, 1e-4)
        
        # Live generation sample with nucleus sampling
        sample_out = generate_sample("Once upon a time, Lily saw a ", max_len=45)
        
        record = {
            "step": step,
            "train_loss": round(step_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
            "tokens_seen": tokens_processed,
            "elapsed_sec": round(elapsed, 1),
            "tok_per_sec": round(tok_rate, 1),
            "generated_sample": sample_out
        }
        history.append(record)
        
        print(f"  Step {step:>4}/{TOTAL_STEPS} | Train Loss: {step_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Speed: {tok_rate:.1f} tok/s")
        print(f"    --> Generation Preview: \"{sample_out}\"")
        model.train()

# 5. Save Checkpoint & Results
print(f"\n[4/4] Saving Final Weights & Verification Artifacts...")
ckpt_path = CHECKPOINT_DIR / "iveri_hyperion_real_english.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "history": history,
    "params": param_count,
    "config": {"hidden_dim": 406, "num_layers": 4, "memory_dim": 64}
}, ckpt_path)
print(f"  [SAVED] Checkpoint saved: {ckpt_path.name}")

results_path = REPORTS_DIR / "iveri_hyperion_real_english_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": "IVERI-HYPERION",
        "paradigm": "ENSN (Entropy-Driven Neural State Network)",
        "params": param_count,
        "dataset": "TinyStories (Real English Text)",
        "final_val_loss": history[-1]["val_loss"],
        "final_val_ppl": history[-1]["val_perplexity"],
        "history": history
    }, f, indent=2)
print(f"  [SAVED] Results logged: {results_path.name}")

# Generate Learning Curve Plot
plt.figure(figsize=(8.5, 4.5), dpi=300)
steps = [h["step"] for h in history]
val_losses = [h["val_loss"] for h in history]
train_losses = [h["train_loss"] for h in history]

plt.plot(steps, train_losses, 'b--', label="Training Loss (Next-Byte Cross-Entropy)")
plt.plot(steps, val_losses, 'g-o', linewidth=2.0, label="Validation Loss (Real English Held-Out)")
plt.xlabel("Training Steps", fontsize=11, fontweight='bold')
plt.ylabel("Cross-Entropy Loss", fontsize=11, fontweight='bold')
plt.title("IVERI-HYPERION: Real English Training Dynamics (TinyStories)", fontsize=12, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

plot_out = current_dir / "paper1_real_english_loss_curves.png"
plt.savefig(plot_out)
plt.close()
print(f"  [SAVED] Learning curve chart: {plot_out.name}")

print("\n" + "=" * 80)
print(f"  [PHASE 3 COMPLETE] IVERI-HYPERION Real English Training Finished Successfully!")
print(f"  Final Validation Perplexity on Real English: {history[-1]['val_perplexity']}")
print("=" * 85)
