#!/usr/bin/env python
# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Run static type checking on the IVERI CORE codebase."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Run type check using mypy."""
    print("Running Type Checking...")

    try:
        res = subprocess.run(
            [sys.executable, "-m", "mypy", "."],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print("\033[92mPASSED: Type checking clean\033[0m")
            return 0
        else:
            print("\033[91mFAILED: Type checking errors found:\033[0m")
            print(res.stdout)
            return 1
    except FileNotFoundError:
        print("\033[93mSKIPPED: Type checking (Mypy is not installed)\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(main())
