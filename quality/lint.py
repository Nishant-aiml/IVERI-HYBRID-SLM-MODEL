#!/usr/bin/env python
# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run linting checks on the IVERI CORE codebase."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
    "configs",
    "core",
    "model",
    "data",
    "training",
    "evaluation",
    "baselines",
    "utils",
    "scripts",
    "tests",
    "quality",
]


def run_py_compile() -> bool:
    """Fallback: checks syntax of all Python files in SOURCE_DIRS."""
    print("Mypy/Ruff not available, running fallback py_compile checks...")
    success = True
    for folder in SOURCE_DIRS:
        dir_path = PROJECT_ROOT / folder
        if not dir_path.exists():
            continue
        for file_path in dir_path.rglob("*.py"):
            try:
                py_compile.compile(str(file_path), doraise=True)
            except py_compile.PyCompileError as e:
                print(f"Syntax Error in {file_path.relative_to(PROJECT_ROOT)}:\n{e}")
                success = False
    return success


def main() -> int:
    """Run lint checks using ruff, or syntax check as fallback."""
    print("Running Linting Checks...")

    # Check for ruff
    try:
        res = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("\033[92mPASSED: Linting checks clean\033[0m")
            return 0
        else:
            print("\033[91mFAILED: Linting checks found issues:\033[0m")
            print(res.stdout)
            print(res.stderr)
            return 1
    except FileNotFoundError:
        # Ruff not installed, fallback to compiling syntax
        if run_py_compile():
            print("\033[93mPASSED: Code compiles successfully (Ruff skipped)\033[0m")
            return 0
        else:
            print("\033[91mFAILED: Syntax compilation errors found\033[0m")
            return 1


if __name__ == "__main__":
    sys.exit(main())
