# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.x Architecture Regression Suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE, PAD_BYTE
from model.iveri_core import IVERIModel
from inference.engine import InferenceEngine
from training.trainer import Trainer



class MockPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len: int = 16) -> None:
        self.seq_len = seq_len

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        return x, y


def run_regression() -> bool:
    print("======================================================================")
    print("IVERI Architecture Regression Suite -- Running Checks...")
    print("======================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    
    config = IVERIConfig()
    # Micro config for speed in test
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.mamba_ratio = 1
    config.model.num_experts = 2
    config.model.num_active_experts = 1
    config.model.max_recursion_depth = 2
    config.model.titans_memory_dim = 8
    
    config.training.seq_len = 16
    config.training.batch_size = 2
    config.training.gradient_accumulation = 1
    config.hardware.device = str(device)
    config.hardware.mixed_precision = "fp32"
    
    # ── Check 1: Imports ──────────────────────────────────────────────────
    print("[1/6] Running Import Health Check...")
    from model.iveri_core import IVERIModel
    from baselines import BaselineTransformer, TinyMamba
    from training.pretrain_runner import run_pretraining
    from training.sft_runner import run_sft
    from inference.engine import InferenceEngine

    print("Imports: OK")

    # ── Check 2: Forward & Backward ──────────────────────────────────────
    print("[2/6] Running Forward/Backward pass on micro model...")
    model = IVERIModel(config)
    model.to(device)
    model.train()
    
    x = torch.randint(0, 256, (2, 16), device=device)
    outputs = model(x, return_dict=True)
    logits = outputs["logits"]
    assert logits.shape == (2, 16, BYTE_VOCAB_SIZE), f"Logits shape mismatch: {logits.shape}"
    assert not torch.isnan(logits).any().item(), "Logits contain NaNs!"
    
    loss = logits.mean() + outputs["aux_loss"]
    loss.backward()
    print("Forward & Backward: OK")

    # ── Check 3: Gradient Flow ───────────────────────────────────────────
    print("[3/6] Checking Gradient Flow...")
    grad_norms = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms.append(p.grad.norm().item())
            
    assert len(grad_norms) > 0, "No parameters received gradients!"
    assert all(torch.isfinite(torch.tensor(g)).item() for g in grad_norms), "NaN/Inf in gradients!"
    print(f"Gradient flow verified: {len(grad_norms)} active gradient tensors.")

    # ── Check 4: Checkpoint Save/Load Round-Trip ─────────────────────────
    print("[4/6] Verifying Checkpoint Save/Load...")
    model.eval()
    with torch.no_grad():
        ref_logits = model(x, return_dict=False)
        
    chk_dir = Path("scratch")
    chk_dir.mkdir(exist_ok=True)
    chk_path = chk_dir / "regression_ckpt.pt"
    
    model.save_checkpoint(chk_path, step=0)
    
    model_loaded = IVERIModel(config)
    model_loaded.to(device)
    model_loaded.load_checkpoint(chk_path)
    model_loaded.eval()
    
    with torch.no_grad():
        loaded_logits = model_loaded(x, return_dict=False)
        
    if chk_path.exists():
        os.remove(chk_path)
        
    max_diff = (ref_logits - loaded_logits).abs().max().item()
    assert max_diff < 1e-6, f"Checkpoint restoration logit difference too high: {max_diff:.2e}"
    print(f"Checkpoint round-trip matched. Max diff: {max_diff:.2e}")

    # ── Check 5: Inference Generation ────────────────────────────────────
    print("[5/6] Verifying Inference Engine & Generation...")
    engine = InferenceEngine(model, device=device)


    prompt = "Once"
    result = engine.generate(prompt, max_new_tokens=5)
    out_text = result.text
    safe_text = "".join(c if ord(c) < 128 else "?" for c in out_text)
    print(f"Generated text: '{safe_text}'")
    assert isinstance(out_text, str), "Generated output must be a string!"



    print("Inference generation: OK")

    # ── Check 6: Single Training Step ────────────────────────────────────
    print("[6/6] Verifying Single Training Step via Trainer...")
    ds = MockPretrainDataset(seq_len=16)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    
    trainer = Trainer(model, config, train_dataloader=loader)
    trainer.optimizer.zero_grad(set_to_none=True)
    
    batch = next(iter(loader))
    tx, ty = batch
    tx = tx.to(device)
    ty = ty.to(device)
    
    outputs = model(tx, return_dict=True)
    logits = outputs["logits"]
    loss_ce = torch.nn.functional.cross_entropy(logits.view(-1, BYTE_VOCAB_SIZE), ty.view(-1))
    loss_comp = loss_ce + 0.01 * outputs["aux_loss"]
    
    loss_comp.backward()
    trainer.optimizer.step()
    print("Trainer single step: OK")
    
    print("======================================================================")
    print("IVERI Architecture Regression Suite: ALL CHECKS PASSED.")
    print("======================================================================")
    return True


if __name__ == "__main__":
    try:
        success = run_regression()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
