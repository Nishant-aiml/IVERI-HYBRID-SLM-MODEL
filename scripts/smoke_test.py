# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.1 Forward Pass & Checkpoint Save/Load Smoke Test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Ensure path is correct
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE
from model.iveri_core import IVERIModel



def run_smoke_test() -> bool:
    print("Initializing Phase 7.1 Smoke Test...")

    # 1. Configuration
    config = IVERIConfig()
    device = torch.device(config.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Model Instantiation
    model = IVERIModel(config)
    model.to(device)
    model.eval()
    print("Model initialized.")

    # 3. Forward Pass Smoke Test
    B, S = 2, 64
    raw_bytes = torch.randint(0, 256, (B, S), device=device)
    print(f"Input shape: {raw_bytes.shape}")

    with torch.no_grad():
        outputs = model(raw_bytes, return_dict=True)

    # 4. Verify Outputs
    assert isinstance(outputs, dict), "Output must be a dictionary when return_dict=True"
    print("Output format: Dict [OK]")

    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape} (Expected: ({B}, {S}, {BYTE_VOCAB_SIZE}))")
    assert logits.shape == (B, S, BYTE_VOCAB_SIZE), f"Expected shape ({B}, {S}, {BYTE_VOCAB_SIZE}), got {logits.shape}"


    
    # Check for NaNs
    has_nan = torch.isnan(logits).any().item()
    print(f"Has NaNs: {has_nan}")
    assert not has_nan, "Logits contain NaNs!"

    # Check aux_loss
    aux_loss = outputs.get("aux_loss")
    print(f"Aux loss: {aux_loss}")
    assert aux_loss is not None, "aux_loss is missing"
    assert torch.isfinite(aux_loss).item(), f"aux_loss is not finite: {aux_loss}"

    # Check telemetry
    telemetry = outputs.get("telemetry")
    print(f"Telemetry: {telemetry}")
    assert isinstance(telemetry, dict), "telemetry must be a dict"
    assert "forward_latency_seconds" in telemetry or "end_to_end_forward_latency_seconds" in telemetry, "missing latency in telemetry"

    # 5. Backward Pass Check
    model.train()
    outputs_train = model(raw_bytes, return_dict=True)
    loss = outputs_train["logits"].mean() + outputs_train["aux_loss"]
    print(f"Mock Train Loss: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
            else:
                print(f"WARNING: Parameter {name} has no gradient!")

    print(f"Parameters with gradients: {len(grad_norms)}/{len([p for p in model.parameters() if p.requires_grad])}")
    assert len(grad_norms) > 0, "No gradients were generated during backward!"
    assert all(torch.isfinite(torch.tensor(g)).item() for g in grad_norms), "Some gradients are NaN or Inf!"
    print("Backward pass & gradients: [OK]")

    # 6. Checkpoint Save/Load Round-Trip
    print("Running checkpoint round-trip validation...")
    model.eval()
    with torch.no_grad():
        ref_logits = model(raw_bytes, return_dict=False)

    checkpoint_dir = Path("scratch")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "smoke_ckpt.pt"

    # Save
    model.save_checkpoint(checkpoint_path, step=0)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Load into new instance
    model_loaded = IVERIModel(config)
    model_loaded.to(device)
    model_loaded.load_checkpoint(checkpoint_path)
    model_loaded.eval()
    print("Checkpoint loaded into new model instance.")

    with torch.no_grad():
        loaded_logits = model_loaded(raw_bytes, return_dict=False)

    # Compare
    max_diff = (ref_logits - loaded_logits).abs().max().item()
    print(f"Max logit difference: {max_diff:.2e}")
    
    # Cleanup
    if checkpoint_path.exists():
        os.remove(checkpoint_path)

    assert max_diff < 1e-6, f"Checkpoint mismatch! max_diff={max_diff:.2e}"
    print("Checkpoint round-trip bitwise match: [OK]")

    print("ALL SMOKE TESTS PASSED successfully.")
    return True


if __name__ == "__main__":
    try:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
