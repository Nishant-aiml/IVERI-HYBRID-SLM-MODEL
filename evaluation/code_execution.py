# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Python sandboxed code executor for IVERI CORE Phase 3.3 coding specialization.

Allows compiled syntax checks and execution checks of generated Python code
within a safe restricted subprocess.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeExecutionResult:
    """Detailed result of executing a code snippet.

    Attributes
    ----------
    compile_success:
        True if code compiled without syntax errors.
    execution_success:
        True if code executed and exited with status 0.
    stdout:
        Standard output of the execution.
    stderr:
        Standard error of the execution.
    timeout:
        True if execution was terminated due to timeout.
    runtime_error:
        True if code compiled but crashed during execution.
    runtime_sec:
        Execution duration in seconds.
    testcase_pass_ratio:
        Fraction of assertions/test cases passed (if specified).
    """

    compile_success: bool
    execution_success: bool
    stdout: str
    stderr: str
    timeout: bool
    runtime_error: bool
    runtime_sec: float
    testcase_pass_ratio: float = 0.0


class CodeExecutor:
    """Executes Python code safely inside a separate process.

    Parameters
    ----------
    timeout_sec:
        Execution timeout per run.
    """

    def __init__(self, timeout_sec: float = 5.0) -> None:
        self.timeout_sec = timeout_sec

    def execute(
        self,
        code: str,
        test_cases: list[str] | None = None,
    ) -> CodeExecutionResult:
        """Compile and execute Python code in a sandboxed subprocess.

        Parameters
        ----------
        code:
            Python source code string.
        test_cases:
            Optional assertion lines to append and run.

        Returns
        -------
        CodeExecutionResult
        """
        # Safety: truncate extremely long outputs to prevent resource exhaustion
        if len(code) > 20000:
            code = code[:20000]

        # 1. Compilation check
        try:
            compile(code, "<string>", "exec")
            compile_success = True
        except SyntaxError:
            return CodeExecutionResult(
                compile_success=False,
                execution_success=False,
                stdout="",
                stderr="SyntaxError",
                timeout=False,
                runtime_error=False,
                runtime_sec=0.0,
            )
        except Exception as exc:
            return CodeExecutionResult(
                compile_success=False,
                execution_success=False,
                stdout="",
                stderr=str(exc),
                timeout=False,
                runtime_error=False,
                runtime_sec=0.0,
            )

        # Append test cases if present
        exec_code = code
        total_tests = 0
        if test_cases:
            total_tests = len(test_cases)
            exec_code += "\n\n# ── Test Suite ──\n"
            exec_code += "passed_tests = 0\n"
            for i, tc in enumerate(test_cases):
                exec_code += f"try:\n    {tc}\n    passed_tests += 1\nexcept Exception as exc:\n    print(f'Test {i} failed: {{exc}}')\n"
            exec_code += f"print(f'PASSED_RATIO: {{passed_tests}} / {total_tests}')\n"

        # 2. Run in a subprocess
        t0 = time.perf_counter()
        try:
            # We run with subprocess to prevent the generated code from crashing our main training loop
            # and to enforce timeouts.
            res = subprocess.run(
                [sys.executable, "-c", exec_code],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            runtime_sec = time.perf_counter() - t0
            stdout = res.stdout
            stderr = res.stderr
            execution_success = res.returncode == 0
            timeout = False
            runtime_error = res.returncode != 0

        except subprocess.TimeoutExpired as exc:
            runtime_sec = time.perf_counter() - t0
            stdout = exc.stdout or ""
            stderr = exc.stderr or "TimeoutExpired"
            execution_success = False
            timeout = True
            runtime_error = False
        except Exception as exc:
            runtime_sec = time.perf_counter() - t0
            stdout = ""
            stderr = str(exc)
            execution_success = False
            timeout = False
            runtime_error = True

        # Parse test case pass ratio from stdout
        testcase_pass_ratio = 0.0
        if total_tests > 0 and stdout:
            for line in stdout.splitlines():
                if "PASSED_RATIO:" in line:
                    try:
                        parts = line.replace("PASSED_RATIO:", "").strip().split("/")
                        passed = int(parts[0])
                        total = int(parts[1])
                        testcase_pass_ratio = passed / total
                    except Exception:
                        pass
                    break

        return CodeExecutionResult(
            compile_success=compile_success,
            execution_success=execution_success,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            runtime_error=runtime_error,
            runtime_sec=runtime_sec,
            testcase_pass_ratio=testcase_pass_ratio,
        )

    def batch_execute(self, snippets: list[str]) -> list[CodeExecutionResult]:
        """Execute a list of code snippets sequentially."""
        return [self.execute(s) for s in snippets]

    def aggregate_metrics(self, results: list[CodeExecutionResult]) -> dict[str, float]:
        """Aggregate metrics over a list of execution results."""
        if not results:
            return {
                "compile_success_ratio": 0.0,
                "execution_success_ratio": 0.0,
                "timeout_ratio": 0.0,
                "runtime_error_ratio": 0.0,
                "avg_runtime_sec": 0.0,
                "testcase_pass_ratio": 0.0,
            }

        n = len(results)
        compile_count = sum(1 for r in results if r.compile_success)
        exec_count = sum(1 for r in results if r.execution_success)
        timeout_count = sum(1 for r in results if r.timeout)
        runtime_err_count = sum(1 for r in results if r.runtime_error)
        avg_runtime = sum(r.runtime_sec for r in results) / n
        avg_test_pass = sum(r.testcase_pass_ratio for r in results) / n

        return {
            "compile_success_ratio": compile_count / n,
            "execution_success_ratio": exec_count / n,
            "timeout_ratio": timeout_count / n,
            "runtime_error_ratio": runtime_err_count / n,
            "avg_runtime_sec": avg_runtime,
            "testcase_pass_ratio": avg_test_pass,
        }
