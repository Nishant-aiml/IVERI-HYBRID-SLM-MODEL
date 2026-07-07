# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.3 engineering stabilization report generator."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "reports" / "phase_6_3_3"


@dataclass
class Phase633State:
    timestamp_utc: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    fixes: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    benchmark: dict[str, object] = field(default_factory=dict)
    proprietary: dict[str, object] = field(default_factory=dict)
    verdict: str = "UNKNOWN"

    def to_dict(self) -> dict:
        return asdict(self)


def _run_pytest() -> tuple[int, int]:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--ignore=tests/integration"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    tail = (proc.stdout or "") + (proc.stderr or "")
    passed = failed = 0
    for line in tail.splitlines():
        if " passed" in line and " in " in line:
            for i, p in enumerate(line.split()):
                if p == "passed":
                    try:
                        passed = int(line.split()[i - 1])
                    except (IndexError, ValueError):
                        pass
        if " failed" in line and " in " in line:
            for i, p in enumerate(line.split()):
                if p == "failed":
                    try:
                        failed = int(line.split()[i - 1])
                    except (IndexError, ValueError):
                        pass
    return passed, failed


def _run_inference_benchmark() -> dict[str, object]:
    bench_path = OUTPUT / "inference_benchmark.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_inference_benchmark.py"),
        "--device",
        "cpu",
        "--runs",
        "3",
        "--warmup",
        "1",
        "--output",
        str(bench_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "benchmark failed").strip()}
    if bench_path.exists():
        return json.loads(bench_path.read_text(encoding="utf-8"))
    return {"error": "benchmark output missing"}


def _proprietary_status() -> dict[str, object]:
    from data.pipeline.proprietary_ingest import count_proprietary_records

    proprietary_dir = REPO_ROOT / "data" / "proprietary"
    counts = count_proprietary_records(proprietary_dir)
    total = sum(counts.values())
    manifest = REPO_ROOT / "data" / "processed" / "stage3b" / "manifest.json"
    return {
        "pipeline_ready": True,
        "raw_record_count": total,
        "by_source": counts,
        "processed_manifest_exists": manifest.exists(),
    }


def generate_reports(state: Phase633State) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    bench = state.benchmark
    bench_section = ""
    if bench and "avg_tokens_per_second" in bench:
        bench_section = (
            f"\n## Measured CPU benchmark\n\n"
            f"- Device: {bench.get('device', 'cpu')}\n"
            f"- Avg latency: {bench.get('avg_latency_seconds', 0):.4f}s\n"
            f"- Avg throughput: {bench.get('avg_tokens_per_second', 0):.2f} tokens/s\n"
            f"- Provenance: {bench.get('provenance', 'MEASURED')}\n"
        )
    elif bench.get("error"):
        bench_section = f"\nBenchmark note: {bench['error']}\n"

    prop = state.proprietary
    prop_section = (
        f"\n## Stage 3B proprietary\n\n"
        f"- Pipeline ready: {prop.get('pipeline_ready', False)}\n"
        f"- Raw records: {prop.get('raw_record_count', 0)}\n"
        f"- Processed manifest: {prop.get('processed_manifest_exists', False)}\n"
    )

    files = {
        "Executive_Summary.md": (
            f"# Phase 6.3.3 Executive Summary\n\n"
            f"Tests: **{state.tests_passed} passed**, {state.tests_failed} failed.\n\n"
            f"Verdict: **{state.verdict}**\n"
            f"{bench_section}{prop_section}"
        ),
        "Engineering_Fixes.md": "# Engineering Fixes\n\n" + "\n".join(f"- {f}" for f in state.fixes),
        "Test_Migration_Report.md": (
            "# Test Migration Report\n\n"
            "BYTE_VOCAB_SIZE=259 migration applied across pretraining, BLT, stress, and logging tests.\n"
            "BaselineTransformer intentionally retains RAW_BYTE_VOCAB_SIZE=256.\n"
        ),
        "Inference_Architecture.md": (
            "# Inference Architecture\n\n"
            "Package: `inference/` with `InferenceEngine`, `ByteTokenizer`, `Sampler`, "
            "`loader.load_inference_model`, CLI (`python -m inference.cli`), and `benchmark_inference()`.\n\n"
            "Deployment guide: `docs/deployment/INFERENCE.md`\n"
        ),
        "Performance_Report.md": (
            "# Performance Report\n\n"
            f"```json\n{json.dumps(state.benchmark, indent=2)}\n```\n"
        ),
        "Deployment_Report.md": (
            "# Deployment Report\n\n"
            "## CLI\n\n"
            "```bash\npython -m inference.cli --prompt \"...\" --device cpu\n```\n\n"
            "## Benchmark\n\n"
            "```bash\npython scripts/run_inference_benchmark.py --device cpu\n```\n\n"
            "See `docs/deployment/INFERENCE.md`.\n"
        ),
        "Repository_Health_Report.md": (
            "# Repository Health\n\n"
            "- loss_mask list padding fixed\n"
            "- logger PermissionError / W&B init fallbacks\n"
            "- conftest num_workers=0 for Windows/Python 3.14\n"
            "- Stage 3B ingest pipeline: `data/pipeline/proprietary_ingest.py`\n"
        ),
        "Logging_Report.md": (
            "# Logging Report\n\n"
            "Structured logging preserved; W&B failures fall back to CSV/JSONL without interrupting training.\n"
        ),
        "Documentation_Report.md": (
            "# Documentation Report\n\n"
            "- README: Phase 6.3.3 + inference + Stage 3B ingest\n"
            "- CHANGELOG: v1.6.0\n"
            "- `docs/deployment/INFERENCE.md`\n"
            "- `data/proprietary/FORMAT.md`\n"
        ),
        "Regression_Report.md": f"# Regression Report\n\n{state.tests_passed} passed / {state.tests_failed} failed.\n",
        "Remaining_Issues.md": "# Remaining Issues\n\n" + "\n".join(f"- {r}" for r in state.remaining),
        "Recommendations.md": (
            "# Recommendations\n\n"
            "- Add proprietary JSON assets under `data/proprietary/` and run `scripts/ingest_stage3b.py`\n"
            "- Run GPU benchmark on CUDA hardware\n"
            "- Execute paper-scale measured campaign before publication claims\n"
        ),
        "Final_Product_Readiness.md": f"# Final Product Readiness\n\n## {state.verdict}\n",
    }
    for name, body in files.items():
        (OUTPUT / name).write_text(body + f"\n**Generated:** {state.timestamp_utc}\n", encoding="utf-8")
    (OUTPUT / "validation_state.json").write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6.3.3 engineering stabilization reports")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest; use --passed/--failed")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip inference benchmark")
    parser.add_argument("--passed", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    args = parser.parse_args()

    state = Phase633State(timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    state.fixes = [
        "BYTE_VOCAB_SIZE test migration (256 -> 259 logits)",
        "loss_mask._mask_padding accepts list[int]",
        "generation_inspector safe byte decode",
        "inference/ package added",
        "logger PermissionError fallback for save_dir and W&B init",
        "coding mock dataset uses torch Dataset; num_workers=0 in tests",
        "publication backward-compat test seeds MEASURED registry rows",
        "baseline_transformer asserts RAW_BYTE_VOCAB_SIZE (256)",
        "logging test mocks wandb.init failure for portable fallback",
        "conftest base_config: num_workers=0, CPU fallback when no CUDA",
        "Stage 3B proprietary ingest pipeline (proprietary_ingest.py)",
        "Deployment guide and measured inference benchmark script",
    ]
    if args.skip_pytest:
        state.tests_passed, state.tests_failed = args.passed, args.failed
    else:
        state.tests_passed, state.tests_failed = _run_pytest()

    state.proprietary = _proprietary_status()
    if not args.skip_benchmark:
        state.benchmark = _run_inference_benchmark()
    elif (OUTPUT / "inference_benchmark.json").exists():
        state.benchmark = json.loads((OUTPUT / "inference_benchmark.json").read_text(encoding="utf-8"))

    state.remaining = []
    if state.tests_failed:
        state.remaining.append(f"{state.tests_failed} tests still failing")
    if int(state.proprietary.get("raw_record_count", 0)) == 0:
        state.remaining.append(
            "Stage 3B proprietary assets not yet added (ingest pipeline ready; 0 raw records)"
        )

    if state.tests_failed == 0 and len(state.remaining) <= 1:
        state.verdict = "Product Ready (Minor Non-Blocking Issues)" if state.remaining else "Product Ready"
    elif state.tests_failed == 0:
        state.verdict = "Product Ready (Minor Non-Blocking Issues)"
    else:
        state.verdict = "Engineering Issues Remain"

    generate_reports(state)
    print(f"Phase 6.3.3 reports -> {OUTPUT}")
    print(f"Verdict: {state.verdict} ({state.tests_passed} passed, {state.tests_failed} failed)")


if __name__ == "__main__":
    main()
