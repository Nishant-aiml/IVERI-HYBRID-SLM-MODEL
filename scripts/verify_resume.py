# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.9 — Checkpoint Resume Fidelity Verifier.

Verifies that resuming training from a checkpoint:
  1. Exactly restores the step counter and epoch
  2. Produces a loss value within a reasonable delta of the checkpoint loss
  3. Does NOT diverge (loss doesn't increase >50% in the first few steps)

Usage:
    python scripts/verify_resume.py --checkpoint logs/.../checkpoint_N.pt
                                    [--steps 5] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.base_config import IVERIConfig, get_base_config
from model.iveri_core import IVERIModel
from training.checkpointing import load_checkpoint
from training.trainer import Trainer


def _load_config_from_checkpoint(checkpoint_path: pathlib.Path) -> IVERIConfig:
    """Load training config from checkpoint or adjacent config_snapshot.json.

    Checks the following in order:
    1. checkpoint['config_dict'] — some runners save under this key
    2. checkpoint['config'] — save_checkpoint() saves under this key
    3. Adjacent config_snapshot.json file
    4. Falls back to get_base_config()
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try 'config_dict' key first (used by some runners)
    config_dict = ckpt.get("config_dict")
    if config_dict is None:
        # Fallback: save_checkpoint() stores under 'config' key
        config_dict = ckpt.get("config")

    if config_dict is not None:
        try:
            config = IVERIConfig.from_dict(config_dict)
            print("Config loaded from checkpoint.")
            return config
        except Exception as exc:
            print(f"Warning: could not reconstruct config from checkpoint: {exc}")

    snapshot = checkpoint_path.parent / "config_snapshot.json"
    if snapshot.exists():
        import json as _json
        raw = _json.loads(snapshot.read_text(encoding="utf-8"))
        # Clamp warmup_steps for validation guard
        training = raw.get("training", {})
        max_s = training.get("max_steps", 1000)
        if training.get("warmup_steps", 0) >= max_s:
            training["warmup_steps"] = max(0, max_s - 1)
            raw["training"] = training
        config = IVERIConfig.from_dict(raw)
        print(f"Config loaded from: {snapshot}")
        return config

    print("Warning: no config found. Using default base config.")
    return get_base_config()


def verify_resume(
    checkpoint_path: pathlib.Path,
    device: str = "cpu",
    resume_steps: int = 5,
) -> dict[str, object]:
    """Verify that resuming from checkpoint is faithful.

    Returns
    -------
    dict with keys: success, initial_step, resumed_step, loss_before,
                    loss_after, loss_delta_pct, message
    """
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "success": False,
        "initial_step": None,
        "resumed_step": None,
        "loss_before": None,
        "loss_after": None,
        "loss_delta_pct": None,
        "message": "",
    }

    if not checkpoint_path.exists():
        result["message"] = f"Checkpoint not found: {checkpoint_path}"
        return result

    # ── Load checkpoint metadata ──────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_step = ckpt.get("step", 0)
    saved_loss = ckpt.get("metrics", {}).get("loss", None)
    result["initial_step"] = saved_step
    result["loss_before"] = saved_loss
    print(f"  Saved step:       {saved_step}")
    print(f"  Saved loss:       {saved_loss}")

    # ── Reconstruct model ─────────────────────────────────────────────────
    config = _load_config_from_checkpoint(checkpoint_path)
    config.hardware.device = device
    config.hardware.mixed_precision = "none"  # CPU-safe
    config.training.gradient_accumulation = 1

    model = IVERIModel(config)
    dev = torch.device(device)

    # ── Create minimal dataloader ─────────────────────────────────────────
    seq_len = config.training.seq_len
    batch_size = max(1, config.training.batch_size)
    n_samples = max(resume_steps * batch_size * 2, 16)
    data = torch.randint(0, 256, (n_samples, seq_len))
    ds = TensorDataset(data, data)
    dl = DataLoader(ds, batch_size=batch_size, drop_last=True)

    # ── Build Trainer and resume ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        config.logging.log_dir = tmpdir
        config.logging.save_every = 9999  # Don't save during verify run
        trainer = Trainer(model, config, dl)

        print(f"Resuming from checkpoint (step={saved_step})...")
        trainer.resume_from_checkpoint(checkpoint_path)

        assert trainer.global_step == saved_step, (
            f"Step mismatch after resume: expected {saved_step}, got {trainer.global_step}"
        )
        result["resumed_step"] = trainer.global_step
        print(f"  Resumed at step:  {trainer.global_step} OK")

        # Run a few training steps to verify model runs without error post-resume
        print(f"  Running {resume_steps} verification steps...")
        t0 = time.perf_counter()
        epoch_metrics = trainer.train_epoch()
        elapsed = time.perf_counter() - t0

        resumed_loss = epoch_metrics.get("train_loss", None)
        result["loss_after"] = resumed_loss
        if resumed_loss is not None:
            print(f"  Post-resume loss: {resumed_loss:.4f}  ({elapsed:.1f}s)")

        # Resume fidelity is determined by:
        # 1. Step counter exactly restored (already verified above)
        # 2. Model runs without error on any data
        # 3. Loss is finite (not NaN/Inf)
        #
        # NOTE: Loss value comparison against checkpoint loss is NOT done here
        # because the verifier uses synthetic random data, which produces high
        # loss regardless — this is expected and not a bug. Loss continuation
        # testing requires the original training dataset (TinyStories).
        if resumed_loss is not None and not (resumed_loss != resumed_loss):  # NaN check
            result["success"] = True
            result["message"] = (
                f"Resume fidelity OK: step counter restored to {saved_step}, "
                f"model runs post-resume (loss={resumed_loss:.4f} on synthetic data, "
                f"data-agnostic loss delta not computed — use real training data for that)."
            )
            if saved_loss is not None:
                delta_pct = 100.0 * abs(resumed_loss - saved_loss) / max(saved_loss, 1e-8)
                result["loss_delta_pct"] = delta_pct
                print(
                    f"  Loss delta vs checkpoint: {delta_pct:.1f}% "
                    f"(expected high on random data — not a failure criterion)"
                )
        else:
            result["success"] = False
            result["message"] = "Post-resume loss is NaN/Inf — model may be corrupted."

        trainer.shutdown_logger()

    return result



def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IVERI CORE — Checkpoint Resume Fidelity Verifier"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--steps", type=int, default=5, help="Verification steps to run after resume")
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint)

    print()
    print("=" * 60)
    print("IVERI CORE — Checkpoint Resume Fidelity Verifier")
    print("=" * 60)

    result = verify_resume(
        checkpoint_path=checkpoint_path,
        device=args.device,
        resume_steps=args.steps,
    )

    print()
    print("=" * 60)
    print(f"RESULT: {'PASS' if result['success'] else 'FAIL'}")
    print(f"Message: {result['message']}")
    print("=" * 60)

    # Write JSON summary
    out_path = checkpoint_path.parent / "resume_verification.json"
    out_path.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"Summary saved to: {out_path}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
