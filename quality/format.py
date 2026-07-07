#!/usr/bin/env python
# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run formatting checks on the IVERI CORE codebase."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Run format check using black."""
    print("Running Formatting Checks...")

    try:
        res = subprocess.run(
            [sys.executable, "-m", "black", "--check", "--diff", "."],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("\033[92mPASSED: Formatting checks clean\033[0m")
            return 0
        else:
            print("\033[91mFAILED: Formatting issues found. Run 'black .' to fix:\033[0m")
            print(res.stdout)
            return 1
    except FileNotFoundError:
        print("\033[93mSKIPPED: Formatting check (Black is not installed)\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(main())
