#!/usr/bin/env python
# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run all quality checks and produce a summary report."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_check(name: str, cmd: list[str]) -> tuple[str, float, str]:
    """Run a single check command, capture output, status, and duration."""
    start_time = time.time()
    try:
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        duration = time.time() - start_time
        if res.returncode == 0:
            # Check if skipped in output
            if name != "Tests" and ("SKIPPED" in res.stdout or "SKIPPED" in res.stderr):
                return "SKIPPED", duration, res.stdout + res.stderr
            return "PASSED", duration, res.stdout + res.stderr
        else:
            return "FAILED", duration, res.stdout + res.stderr
    except FileNotFoundError:
        duration = time.time() - start_time
        return "SKIPPED", duration, f"Tool not installed: {cmd[0]}"


def main() -> int:
    """Run lint, format, typecheck, and tests."""
    parser = argparse.ArgumentParser(description="IVERI CORE Quality Checker")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Save report to reports/phase_0/quality/report.md",
    )
    args = parser.parse_args()

    checks = {
        "Lint": [sys.executable, "quality/lint.py"],
        "Format": [sys.executable, "quality/format.py"],
        "TypeCheck": [sys.executable, "quality/typecheck.py"],
        "Tests": [sys.executable, "-m", "pytest", "tests/"],
    }

    results = {}
    any_failed = False

    print("=" * 60)
    print("IVERI CORE QUALITY ASSURANCE ROUTINE")
    print("=" * 60)

    for name, cmd in checks.items():
        status, duration, log = run_check(name, cmd)
        results[name] = {"status": status, "duration": duration, "log": log}
        print(f"{name:<12}: {status:<8} ({duration:.2f}s)")
        if status == "FAILED":
            any_failed = True

    print("=" * 60)
    summary_status = "FAILED" if any_failed else "PASSED"
    color_code = "\033[91m" if any_failed else "\033[92m"
    print(f"Overall Status: {color_code}{summary_status}\033[0m")
    print("=" * 60)

    if args.report:
        report_dir = PROJECT_ROOT / "reports/phase_0/quality"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.md"

        markdown = [
            "# Quality Assurance Report — Phase 0",
            f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Status:** {summary_status}",
            "\n## Check Summary",
            "\n| Check | Status | Duration |",
            "|---|---|---|",
        ]
        for name, data in results.items():
            markdown.append(f"| {name} | {data['status']} | {data['duration']:.2f}s |")

        markdown.append("\n## Detailed Logs")
        for name, data in results.items():
            markdown.append(f"\n### {name}")
            markdown.append("```")
            markdown.append(str(data["log"]).strip())
            markdown.append("```")

        report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
        print(f"Saved quality report to: {report_path}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
