"""
Passkey Retrieval & Associative Memory Needle-in-a-Haystack Benchmark
======================================================================
Tests the empirical associative recall capacity of the trained IVERI-HYPERION model
(from reports/proof_campaign/iveri_trained_1000.pt) across context lengths:
S in {64, 128, 256, 512, 1024}.

Measures:
1. Exact Passkey Prediction Accuracy (%)
2. Passkey Cross-Entropy Loss
3. Retrieval Degradation Curve over Context Distance
"""

import sys
import random
import json
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.v1.iveri_v1 import IVERIv1Model, BYTE_VOCAB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPORTS_DIR = REPO_ROOT / "reports" / "final_validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = REPO_ROOT / "reports" / "proof_campaign" / "iveri_trained_1000.pt"

print(f"[Loading IVERI-HYPERION checkpoint from {CKPT_PATH}...]", flush=True)
model = IVERIv1Model(hidden_dim=406, num_layers=4, memory_dim=64).to(device)
state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval()

# Filler text pool from TinyStories
DATA_DIR = REPO_ROOT / "data" / "processed"
val_pt = torch.load(DATA_DIR / "val.pt", map_location="cpu", weights_only=False)
filler_bytes = val_pt["inputs"].flatten().tolist()

CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]
NUM_TRIALS_PER_CTX = 20

results = {}

print("\n" + "=" * 90, flush=True)
print("  PASSKEY RETRIEVAL EMPIRICAL CAPACITY AUDIT (IVERI-HYPERION 10M)", flush=True)
print("=" * 90, flush=True)

for ctx_len in CONTEXT_LENGTHS:
    correct_matches = 0
    total_passkey_loss = 0.0
    passkey_bytes_count = 0

    for trial in range(NUM_TRIALS_PER_CTX):
        # Generate 4 random digits as passkey
        key_str = "".join([str(random.randint(0, 9)) for _ in range(4)])
        needle_str = f" The passkey is {key_str}. "
        query_str = " The passkey is "
        needle_bytes = [b for b in needle_str.encode("utf-8")]
        query_bytes  = [b for b in query_str.encode("utf-8")]
        target_bytes = [b for b in key_str.encode("utf-8")]

        # Construct haystack
        # Position needle in the first 25% - 40% of sequence
        prefix_len = int(ctx_len * 0.25)
        haystack_prefix = filler_bytes[trial * 100 : trial * 100 + prefix_len]
        
        middle_filler_len = ctx_len - len(haystack_prefix) - len(needle_bytes) - len(query_bytes) - len(target_bytes)
        if middle_filler_len < 0:
            middle_filler_len = 10
        middle_filler = filler_bytes[(trial + 1) * 200 : (trial + 1) * 200 + middle_filler_len]

        seq = haystack_prefix + needle_bytes + middle_filler + query_bytes
        # Truncate to exact length before target
        target_start_idx = len(seq)
        full_seq = seq + target_bytes

        seq_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = model(seq_tensor, return_dict=False)  # (1, S, BYTE_VOCAB)
        
        # Check predictions for each target byte
        preds = []
        trial_correct = True
        trial_loss = 0.0

        for i, tb in enumerate(target_bytes):
            pred_pos = target_start_idx + i - 1
            pred_logit = logits[0, pred_pos]
            pred_byte = torch.argmax(pred_logit).item()
            preds.append(pred_byte)
            loss_t = F.cross_entropy(pred_logit.unsqueeze(0), torch.tensor([tb], device=device)).item()
            trial_loss += loss_t
            if pred_byte != tb:
                trial_correct = False

        if trial_correct:
            correct_matches += 1
        total_passkey_loss += trial_loss
        passkey_bytes_count += len(target_bytes)

    acc = (correct_matches / NUM_TRIALS_PER_CTX) * 100.0
    avg_loss = total_passkey_loss / passkey_bytes_count
    results[ctx_len] = {
        "context_length": ctx_len,
        "exact_passkey_accuracy_pct": round(acc, 2),
        "mean_target_byte_loss": round(avg_loss, 4),
        "correct_trials": correct_matches,
        "total_trials": NUM_TRIALS_PER_CTX,
    }
    print(f"  Context {ctx_len:>4} bytes: Exact Accuracy = {acc:>6.2f}% | Mean Byte Loss = {avg_loss:.4f} ({correct_matches}/{NUM_TRIALS_PER_CTX} correct)", flush=True)

out_file = REPORTS_DIR / "passkey_retrieval_capacity_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVED] {out_file}", flush=True)
