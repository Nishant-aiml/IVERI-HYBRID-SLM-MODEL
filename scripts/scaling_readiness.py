# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.10 -- Scaling Readiness Validator.

Instantiates every preset configuration (nano -> max), counts parameters,
estimates FLOPs per token, estimates VRAM usage, and runs a short forward
pass to confirm there are no hidden shape assumptions tied to the nano config.

Usage:
    python scripts/scaling_readiness.py [--device cpu] [--report-dir reports/phase_7]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from configs.base_config import IVERIConfig
from model.iveri_core import IVERIModel

PRESETS_DIR = ROOT / "configs" / "presets"
PRESET_FILES = sorted(PRESETS_DIR.glob("*.yaml"))

# ── VRAM estimation constants ──────────────────────────────────────────────────
BYTES_PER_PARAM_FP32 = 4
BYTES_PER_PARAM_FP16 = 2
OPTIMIZER_MULTIPLIER = 3   # Model + gradient + Adam moments (fp32)
ACTIVATION_PER_TOKEN_MB = 0.05  # rough estimate: ~50 KB per token in fp32


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def estimate_flops_per_token(config: IVERIConfig) -> int:
    """Rough FLOPs per forward token (multiply-adds * 2 for FLOP count).

    Uses: 2 * num_layers * 12 * seq_len * hidden_dim^2
    Reference: Kaplan et al. (2020), approximate formula for transformer-like models.
    IVERI uses hybrid blocks so this is a lower-bound estimate.
    """
    D = config.model.hidden_dim
    L = config.model.num_layers
    # Attention-equivalent cost per layer per token
    flops_per_layer = 12 * D * D  # standard estimate
    # MoE factor: only num_active / num_experts fraction of expert FLOPs
    k = config.model.num_active_experts
    e = config.model.num_experts
    moe_factor = k / e
    # Titans memory: approximate as 2 * D^2 extra per layer
    titans_extra = 2 * D * D if config.model.use_titans else 0
    return int(2 * L * (flops_per_layer * (1 + moe_factor) + titans_extra))


def estimate_vram_mb(
    param_count: int,
    config: IVERIConfig,
    batch_size: int = 1,
    dtype: str = "fp16",
) -> dict[str, float]:
    """Estimate VRAM usage at inference and training time (MB)."""
    bytes_per = BYTES_PER_PARAM_FP16 if dtype == "fp16" else BYTES_PER_PARAM_FP32

    model_mb = (param_count * bytes_per) / (1024 ** 2)
    gradient_mb = (param_count * BYTES_PER_PARAM_FP32) / (1024 ** 2)  # Always fp32
    optimizer_mb = gradient_mb * 2  # Adam m + v

    seq_len = config.training.seq_len
    activation_mb = batch_size * seq_len * ACTIVATION_PER_TOKEN_MB

    inference_mb = model_mb + activation_mb
    training_mb = model_mb + gradient_mb + optimizer_mb + activation_mb * 2

    return {
        "model_mb": round(model_mb, 1),
        "gradient_mb": round(gradient_mb, 1),
        "optimizer_mb": round(optimizer_mb, 1),
        "activation_mb": round(activation_mb, 1),
        "inference_total_mb": round(inference_mb, 1),
        "training_total_mb": round(training_mb, 1),
        "inference_gpu_4gb_ok": inference_mb < 4096,
        "training_gpu_8gb_ok": training_mb < 8192,
        "training_gpu_16gb_ok": training_mb < 16384,
        "training_gpu_24gb_ok": training_mb < 24576,
        "training_gpu_40gb_ok": training_mb < 40960,
        "training_gpu_80gb_ok": training_mb < 81920,
    }


def run_forward_pass_check(
    model: torch.nn.Module,
    config: IVERIConfig,
    device: str,
) -> dict[str, object]:
    """Run a minimal forward pass to confirm no shape errors.

    IVERIModel returns a dict: {logits, byte_entropy, patch_entropy,
    boundary_map, aux_loss, telemetry}.
    """
    result: dict[str, object] = {
        "success": False,
        "output_shape": None,
        "has_nan": None,
        "elapsed_ms": None,
        "aux_loss": None,
        "telemetry_keys": None,
        "error": None,
    }
    try:
        model.eval()
        B, S = 1, min(64, config.training.seq_len)  # Keep small for CPU speed
        x = torch.randint(0, 256, (B, S), device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        elapsed = (time.perf_counter() - t0) * 1000

        # IVERIModel returns a dict — extract logits
        if isinstance(out, dict):
            logits = out.get("logits")
            aux_loss = out.get("aux_loss")
            telemetry = out.get("telemetry", {})
            result["telemetry_keys"] = list(telemetry.keys()) if isinstance(telemetry, dict) else []
            if aux_loss is not None:
                result["aux_loss"] = float(aux_loss.item()) if isinstance(aux_loss, torch.Tensor) else float(aux_loss)
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        if logits is None:
            result["error"] = "No 'logits' key in model output dict"
            return result

        result["success"] = True
        result["output_shape"] = list(logits.shape)
        result["has_nan"] = bool(torch.isnan(logits).any())
        result["elapsed_ms"] = round(elapsed, 1)

        # Also verify shape contract: (B, S, vocab_size)
        if len(logits.shape) != 3 or logits.shape[0] != B or logits.shape[1] != S:
            result["has_nan"] = True  # Flag shape contract violation
            result["error"] = f"Unexpected logits shape: {logits.shape}, expected ({B}, {S}, vocab)"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def format_param_count(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def validate_preset(
    preset_path: pathlib.Path,
    device: str,
) -> dict[str, object]:
    """Validate a single preset: load, instantiate, profile, forward pass."""
    result: dict[str, object] = {
        "preset": preset_path.stem,
        "path": str(preset_path),
        "config_load_ok": False,
        "model_init_ok": False,
        "forward_pass": {},
        "params": {},
        "flops_per_token": None,
        "vram": {},
        "issues": [],
        "overall_ok": False,
    }

    # ── 1. Load config ─────────────────────────────────────────────────────────
    try:
        config = IVERIConfig.load(preset_path)
        config.hardware.device = device
        config.hardware.mixed_precision = "none"  # CPU-safe
        result["config_load_ok"] = True
        result["hidden_dim"] = config.model.hidden_dim
        result["num_layers"] = config.model.num_layers
        result["num_heads"] = config.model.num_heads
        result["num_experts"] = config.model.num_experts
        result["num_active_experts"] = config.model.num_active_experts
        result["max_recursion_depth"] = config.model.max_recursion_depth
        result["titans_memory_dim"] = config.model.titans_memory_dim
        result["seq_len"] = config.training.seq_len
        result["batch_size"] = config.training.batch_size
        result["grad_accum"] = config.training.gradient_accumulation
        result["effective_batch"] = config.training.batch_size * config.training.gradient_accumulation
        result["max_steps"] = config.training.max_steps
    except Exception as exc:
        result["issues"].append(f"Config load FAILED: {exc}")
        return result

    # ── 2. Instantiate model ───────────────────────────────────────────────────
    try:
        model = IVERIModel(config).to(device)
        result["model_init_ok"] = True
    except Exception as exc:
        result["issues"].append(f"Model init FAILED: {exc}")
        result["issues"].append(traceback.format_exc())
        return result

    # ── 3. Parameter count ─────────────────────────────────────────────────────
    params = count_parameters(model)
    result["params"] = params
    result["param_count_human"] = format_param_count(params["total"])

    # ── 4. FLOPs estimate ─────────────────────────────────────────────────────
    result["flops_per_token"] = estimate_flops_per_token(config)
    result["gflops_per_token"] = round(result["flops_per_token"] / 1e9, 4)

    # ── 5. VRAM estimate ──────────────────────────────────────────────────────
    result["vram"] = estimate_vram_mb(params["total"], config, batch_size=config.training.batch_size)

    # ── 6. Forward pass ───────────────────────────────────────────────────────
    fwd = run_forward_pass_check(model, config, device)
    result["forward_pass"] = fwd

    # ── 7. Sanity checks ──────────────────────────────────────────────────────
    if fwd.get("has_nan"):
        result["issues"].append("WARNING: Forward pass produced NaN outputs")
    if not fwd.get("success"):
        result["issues"].append(f"Forward pass FAILED: {fwd.get('error')}")

    # Check head divisibility
    if config.model.hidden_dim % config.model.num_heads != 0:
        result["issues"].append(
            f"hidden_dim ({config.model.hidden_dim}) not divisible by "
            f"num_heads ({config.model.num_heads})"
        )

    # Check active <= total experts
    if config.model.num_active_experts > config.model.num_experts:
        result["issues"].append(
            f"num_active_experts ({config.model.num_active_experts}) > "
            f"num_experts ({config.model.num_experts})"
        )

    result["overall_ok"] = (
        result["config_load_ok"]
        and result["model_init_ok"]
        and fwd.get("success", False)
        and not fwd.get("has_nan", True)
        and len(result["issues"]) == 0
    )

    del model
    return result


def write_markdown_report(results: list[dict], report_path: pathlib.Path) -> None:
    """Write a Markdown scaling readiness report."""
    lines = [
        "# Phase 7.10 -- IVERI Scaling Readiness Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Summary",
        "",
        "| Preset | Params | Config | Init | Forward | NaN? | VRAM Train | Issues |",
        "|--------|--------|--------|------|---------|------|-----------|--------|",
    ]

    all_ok = True
    for r in results:
        preset = r["preset"]
        params = r.get("param_count_human", "N/A")
        cfg_ok = "[PASS]" if r.get("config_load_ok") else "[FAIL]"
        init_ok = "[PASS]" if r.get("model_init_ok") else "[FAIL]"
        fwd = r.get("forward_pass", {})
        fwd_ok = "[PASS]" if fwd.get("success") else "[FAIL]"
        nan = "YES" if fwd.get("has_nan") else "no"
        vram = r.get("vram", {})
        vram_train = f"{vram.get('training_total_mb', '?'):.0f} MB" if vram else "N/A"
        issues = str(len(r.get("issues", [])))
        lines.append(f"| {preset} | {params} | {cfg_ok} | {init_ok} | {fwd_ok} | {nan} | {vram_train} | {issues} |")
        if not r.get("overall_ok"):
            all_ok = False

    verdict = "ALL PRESETS PASS" if all_ok else "SOME PRESETS HAVE ISSUES"
    lines += [
        "",
        f"## Verdict: {verdict}",
        "",
        "## Detailed Results",
        "",
    ]

    for r in results:
        lines += [
            f"### {r['preset']}",
            "",
            f"- **Parameters**: {r.get('param_count_human', 'N/A')} total",
            f"  - Trainable: {r.get('params', {}).get('trainable', 0):,}",
            f"- **Architecture**: D={r.get('hidden_dim')} L={r.get('num_layers')} H={r.get('num_heads')} Experts={r.get('num_experts')} (K={r.get('num_active_experts')}) Depth={r.get('max_recursion_depth')} TitansDim={r.get('titans_memory_dim')}",
            f"- **Training**: batch={r.get('batch_size')} x accum={r.get('grad_accum')} = eff_batch={r.get('effective_batch')} | seq={r.get('seq_len')} | steps={r.get('max_steps')}",
            f"- **FLOPs/token**: {r.get('gflops_per_token', 'N/A')} GFLOPs (estimate)",
            "",
            "**VRAM Estimates (FP16 inference, FP32 training):**",
            "",
        ]
        vram = r.get("vram", {})
        if vram:
            lines += [
                f"| Mode | VRAM | GPU Compatibility |",
                f"|------|------|-------------------|",
                f"| Model weights (fp16) | {vram.get('model_mb', 0):.0f} MB | -- |",
                f"| Inference (B=1) | {vram.get('inference_total_mb', 0):.0f} MB | {'4GB OK' if vram.get('inference_gpu_4gb_ok') else '>4GB needed'} |",
                f"| Training (B={r.get('batch_size')}) | {vram.get('training_total_mb', 0):.0f} MB | "
                + ("8GB" if vram.get("training_gpu_8gb_ok") else "16GB" if vram.get("training_gpu_16gb_ok") else "24GB" if vram.get("training_gpu_24gb_ok") else "40GB" if vram.get("training_gpu_40gb_ok") else "80GB+")
                + " GPU |",
            ]

        fwd = r.get("forward_pass", {})
        if fwd:
            status = "[PASS]" if fwd.get("success") else "[FAIL]"
            lines += [
                "",
                f"**Forward Pass**: {status}",
                f"- Output shape: {fwd.get('output_shape', 'N/A')}",
                f"- Has NaN: {fwd.get('has_nan', 'N/A')}",
                f"- Latency (B=1, S=64): {fwd.get('elapsed_ms', 'N/A')} ms",
            ]
            if fwd.get("error"):
                lines.append(f"- Error: `{fwd['error']}`")

        issues = r.get("issues", [])
        if issues:
            lines += ["", "**Issues:**"]
            for iss in issues:
                lines.append(f"- {iss}")
        else:
            lines.append("")
            lines.append("**No issues found.**")

        lines += ["", "---", ""]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="IVERI Phase 7.10 Scaling Readiness")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument(
        "--report-dir",
        default="reports/phase_7",
        help="Directory for reports",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        help="Specific preset stems to check (default: all)",
    )
    args = parser.parse_args()

    report_dir = ROOT / args.report_dir

    print()
    print("=" * 68)
    print("  IVERI CORE Phase 7.10 -- Scaling Readiness Validator")
    print("=" * 68)
    print(f"  Device : {args.device}")
    print(f"  Presets: {PRESETS_DIR}")
    print()

    # Filter presets if specified
    presets_to_check = PRESET_FILES
    if args.presets:
        presets_to_check = [p for p in PRESET_FILES if p.stem in args.presets]

    results: list[dict] = []

    for preset_path in presets_to_check:
        print(f"  Checking: {preset_path.name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = validate_preset(preset_path, args.device)
        elapsed = time.perf_counter() - t0
        result["validation_elapsed_s"] = round(elapsed, 1)

        status = "[PASS]" if result["overall_ok"] else "[FAIL]"
        params = result.get("param_count_human", "?")
        fwd_ms = result.get("forward_pass", {}).get("elapsed_ms", "?")
        print(f"{status}  {params:>8}  fwd={fwd_ms}ms  ({elapsed:.1f}s total)")
        if result.get("issues"):
            for iss in result["issues"]:
                print(f"         ! {iss}")

        results.append(result)

    # Save JSON
    json_path = report_dir / "10_scaling_readiness.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save Markdown
    md_path = report_dir / "10_scaling_readiness.md"
    write_markdown_report(results, md_path)

    # Final summary
    n_ok = sum(1 for r in results if r["overall_ok"])
    n_total = len(results)
    all_ok = n_ok == n_total

    print()
    print("=" * 68)
    for r in results:
        icon = "[PASS]" if r["overall_ok"] else "[FAIL]"
        params = r.get("param_count_human", "?")
        vram = r.get("vram", {})
        train_vram = f"{vram.get('training_total_mb', 0):.0f}MB" if vram else "?"
        print(f"  {icon}  {r['preset']:<20}  {params:>8}  train_vram~{train_vram}")
    print("=" * 68)
    verdict = f"PASS ({n_ok}/{n_total} presets OK)" if all_ok else f"FAIL ({n_ok}/{n_total} passed)"
    print(f"  VERDICT: {verdict}")
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
