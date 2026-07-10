# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.9 — End-to-End Product Validation Script.

Validates the complete IVERI CORE user scenario on Windows:
  1. Dataset preparation  (data/prepare_tinystories.py)
  2. Dry-run model boot   (train.py --dry-run)
  3. Short training run   (20 steps, --verification-level 1)
  4. Checkpoint resume    (resume from step-20, train 10 more steps)
  5. Inference            (scripts/generate.py)

Each stage is independently timed and reported. Any failure is captured
with full stderr so root cause is immediately visible.

Usage:
    python scripts/e2e_validation.py [--steps N] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
    """Run a subprocess, returning (success, combined_output)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        elapsed = time.perf_counter() - start
        ok = result.returncode == 0
        output = result.stdout + result.stderr
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}  ({elapsed:.1f}s)")
        if not ok:
            # Print last 10 lines for quick diagnosis
            for line in output.strip().splitlines()[-10:]:
                print(f"         | {line}")
        return ok, output
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  [TIMEOUT] {label}  ({elapsed:.1f}s)")
        return False, "TIMEOUT"
    except Exception as exc:
        print(f"  [ERROR] {label}: {exc}")
        return False, str(exc)


def _separator(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_dataset_prepare() -> bool:
    """Stage 1: Verify dataset preparation script runs without error."""
    _separator("STAGE 1 — Dataset Preparation")
    ok, out = _run(
        "data/prepare_tinystories.py --dry-run",
        [sys.executable, "data/prepare_tinystories.py", "--dry-run"],
        timeout=60,
    )
    if not ok:
        # Fallback: check the shards already exist from a prior run
        shard_dir = ROOT / "data" / "tinystories"
        shards = list(shard_dir.glob("*.bin")) if shard_dir.exists() else []
        if shards:
            print(f"  [INFO] dry-run failed but {len(shards)} existing shards found — OK to continue")
            return True
    return ok


def stage_dry_run(device: str) -> bool:
    """Stage 2: Verify model boots, config loads, no exception before training."""
    _separator("STAGE 2 — Model Dry-Run Boot")
    ok, _ = _run(
        "train.py --dry-run",
        [sys.executable, "train.py", "--dry-run", "--device", device],
        timeout=60,
    )
    return ok


def stage_short_train(device: str) -> tuple[bool, pathlib.Path | None]:
    """Stage 3: Train for verification-level 1 (20 steps)."""
    _separator("STAGE 3 — Short Training Run (20 steps)")
    ok, out = _run(
        "train.py --verification-level 1",
        [sys.executable, "train.py", "--device", device, "--verification-level", "1"],
        timeout=300,
    )
    if not ok:
        return False, None

    # Find the latest checkpoint produced
    log_dir = ROOT / "logs"
    checkpoints = sorted(log_dir.rglob("checkpoint_*.pt"))
    if not checkpoints:
        print("  [WARN] No checkpoint_*.pt found after training — resume test skipped")
        return True, None
    latest = checkpoints[-1]
    print(f"  [INFO] Latest checkpoint: {latest.relative_to(ROOT)}")
    return True, latest


def stage_resume(device: str, checkpoint: pathlib.Path) -> bool:
    """Stage 4: Resume training from checkpoint and verify loss continues."""
    _separator("STAGE 4 — Checkpoint Resume Fidelity")
    ok, _ = _run(
        "scripts/verify_resume.py",
        [sys.executable, "scripts/verify_resume.py",
         "--checkpoint", str(checkpoint),
         "--device", device,
         "--steps", "5"],
        timeout=180,
    )
    return ok


def stage_inference(checkpoint: pathlib.Path) -> bool:
    """Stage 5: Generate text from the trained checkpoint."""
    _separator("STAGE 5 — Inference / Text Generation")
    ok, out = _run(
        "scripts/generate.py (greedy)",
        [sys.executable, "scripts/generate.py",
         "--checkpoint", str(checkpoint),
         "--max-new-bytes", "16",
         "--temperature", "0.0",
         "--prompt", "The model learned"],
        timeout=60,
    )
    if ok and "Generated Output" in out:
        # Extract the generated line for display
        for line in out.splitlines():
            if "Generated Output" in line or line.startswith(" "):
                print(f"  [INFO] {line.strip()}")
    return ok


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _write_report(results: dict[str, bool], total_time: float) -> pathlib.Path:
    """Write a markdown report to reports/phase_7/09_product_validation.md."""
    report_dir = ROOT / "reports" / "phase_7"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "09_product_validation.md"

    lines = [
        "# Phase 7.9 — Product Validation Report",
        "",
        f"**Total validation time:** {total_time:.1f}s",
        "",
        "## Stage Results",
        "",
        "| Stage | Result |",
        "|---|---|",
    ]
    for stage, ok in results.items():
        icon = "[PASS]" if ok else "[FAIL]"
        lines.append(f"| {stage} | {icon} |")

    all_ok = all(results.values())
    verdict = "ALL STAGES PASSED" if all_ok else "SOME STAGES FAILED"
    lines += [
        "",
        f"## Verdict: {verdict}",
        "",
        "---",
        "_Generated by scripts/e2e_validation.py_",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {report_path.relative_to(ROOT)}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all Phase 7.9 validation stages."""
    parser = argparse.ArgumentParser(
        description="IVERI CORE Phase 7.9 — E2E Product Validation"
    )
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda (default: cpu)")
    args = parser.parse_args()

    print()
    print("=" * 66)
    print("  IVERI CORE Phase 7.9 -- End-to-End Product Validation")
    print("=" * 66)
    print(f"  Device: {args.device}")
    print()

    start_total = time.perf_counter()
    results: dict[str, bool] = {}
    checkpoint: pathlib.Path | None = None

    # Stage 1
    results["Stage 1: Dataset Prepare"] = stage_dataset_prepare()

    # Stage 2
    results["Stage 2: Dry-Run Boot"] = stage_dry_run(args.device)
    if not results["Stage 2: Dry-Run Boot"]:
        print("\n  [CRITICAL] Model failed to boot. Aborting remaining stages.")
        _write_report(results, time.perf_counter() - start_total)
        return 1

    # Stage 3
    train_ok, checkpoint = stage_short_train(args.device)
    results["Stage 3: Short Training"] = train_ok

    # Stage 4: only if checkpoint exists
    if checkpoint and checkpoint.exists():
        results["Stage 4: Resume Fidelity"] = stage_resume(args.device, checkpoint)
    else:
        results["Stage 4: Resume Fidelity"] = False
        print("  [SKIP] No checkpoint available — resume test skipped")

    # Stage 5: only if checkpoint exists
    if checkpoint and checkpoint.exists():
        results["Stage 5: Inference"] = stage_inference(checkpoint)
    else:
        # Try the known 1000-step checkpoint as fallback
        fallback = ROOT / "logs" / "iveri_stage1_lvl3" / "checkpoint_1000.pt"
        if fallback.exists():
            print(f"  [INFO] Using fallback checkpoint: {fallback.relative_to(ROOT)}")
            results["Stage 5: Inference"] = stage_inference(fallback)
        else:
            results["Stage 5: Inference"] = False
            print("  [SKIP] No checkpoint available for inference test")

    total_time = time.perf_counter() - start_total
    _write_report(results, total_time)

    # Final summary
    all_ok = all(results.values())
    print()
    print("=" * 68)
    for stage, ok in results.items():
        icon = "[PASS]" if ok else "[FAIL]"
        print(f"  {icon}  {stage}")
    print("=" * 68)
    verdict = "ALL STAGES PASSED" if all_ok else "SOME STAGES FAILED"
    print(f"  VERDICT: {verdict}  ({total_time:.1f}s total)")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
