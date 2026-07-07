# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Code quality analyzer for IVERI CORE Phase 3.3 coding specialization.

Computes structural metrics on code including cyclomatic complexity, function count,
average function length, comment ratios, docstring ratios, and duplicate code ratios.
Supports radon library when available; falls back to AST parsing and heuristics.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeQualityResult:
    """Metrics representing structural code quality.

    Attributes
    ----------
    cyclomatic_complexity:
        Estimated cyclomatic complexity (branches + 1).
    function_count:
        Number of defined functions/methods.
    avg_function_length:
        Average lines of code per function.
    comment_ratio:
        Ratio of comment lines to total lines of code.
    docstring_ratio:
        Fraction of functions containing docstrings.
    duplicate_ratio:
        Fraction of duplicate lines of code.
    total_lines:
        Total lines of text.
    blank_ratio:
        Ratio of blank lines.
    language:
        Target programming language.
    analysis_method:
        One of ``\"radon\"``, ``\"ast\"``, ``\"heuristic\"``, or ``\"skipped\"``.
    """

    cyclomatic_complexity: float
    function_count: int
    avg_function_length: float
    comment_ratio: float
    docstring_ratio: float
    duplicate_ratio: float
    total_lines: int
    blank_ratio: float
    language: str
    analysis_method: str


class CodeQualityAnalyzer:
    """Analyzes source code for complexity and documentation metrics."""

    def analyze(self, code: str, language: str = "python") -> CodeQualityResult:
        """Analyze a string of source code.

        Parameters
        ----------
        code:
            Source code string.
        language:
            Normalised language name.

        Returns
        -------
        CodeQualityResult
        """
        lang = (language or "python").lower().strip()
        lines = code.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return CodeQualityResult(
                cyclomatic_complexity=1.0,
                function_count=0,
                avg_function_length=0.0,
                comment_ratio=0.0,
                docstring_ratio=0.0,
                duplicate_ratio=0.0,
                total_lines=0,
                blank_ratio=0.0,
                language=lang,
                analysis_method="skipped",
            )

        # 1. Blank lines and line duplicate ratio (Feedback #6)
        blank_lines = sum(1 for line in lines if not line.strip())
        blank_ratio = blank_lines / total_lines

        # Calculate duplicate ratio
        non_blank_lines = [l.strip() for l in lines if l.strip()]
        if non_blank_lines:
            unique_lines = set(non_blank_lines)
            duplicate_ratio = 1.0 - (len(unique_lines) / len(non_blank_lines))
        else:
            duplicate_ratio = 0.0

        # Dispatch by language
        if lang == "python":
            return self._analyze_python(code, lines, total_lines, blank_ratio, duplicate_ratio)
        else:
            return self._analyze_heuristic(code, lines, total_lines, blank_ratio, duplicate_ratio, lang)

    def analyze_bytes(self, code_bytes: bytes, language: str = "python") -> CodeQualityResult:
        """Decode and analyze source code bytes."""
        return self.analyze(code_bytes.decode("utf-8", errors="replace"), language)

    def aggregate(self, results: list[CodeQualityResult]) -> dict[str, float]:
        """Aggregate quality metrics over a batch of results."""
        if not results:
            return {
                "avg_cyclomatic_complexity": 0.0,
                "avg_function_count": 0.0,
                "avg_function_length": 0.0,
                "avg_comment_ratio": 0.0,
                "avg_docstring_ratio": 0.0,
                "avg_duplicate_ratio": 0.0,
                "avg_total_lines": 0.0,
                "avg_blank_ratio": 0.0,
            }

        n = len(results)
        return {
            "avg_cyclomatic_complexity": sum(r.cyclomatic_complexity for r in results) / n,
            "avg_function_count": sum(r.function_count for r in results) / n,
            "avg_function_length": sum(r.avg_function_length for r in results) / n,
            "avg_comment_ratio": sum(r.comment_ratio for r in results) / n,
            "avg_docstring_ratio": sum(r.docstring_ratio for r in results) / n,
            "avg_duplicate_ratio": sum(r.duplicate_ratio for r in results) / n,
            "avg_total_lines": sum(r.total_lines for r in results) / n,
            "avg_blank_ratio": sum(r.blank_ratio for r in results) / n,
        }

    # ── Private methods ────────────────────────────────────────────────

    def _analyze_python(
        self,
        code: str,
        lines: list[str],
        total_lines: int,
        blank_ratio: float,
        duplicate_ratio: float,
    ) -> CodeQualityResult:
        """Analyze Python code using AST or radon library."""
        # Try radon first if available
        try:
            from radon.complexity import cc_visit  # type: ignore[import]
            blocks = cc_visit(code)
            if blocks:
                cyclomatic = sum(b.complexity for b in blocks) / len(blocks)
                function_count = sum(1 for b in blocks if b.letter in ("F", "M"))
            else:
                cyclomatic = 1.0
                function_count = 0
            method = "radon"
        except ImportError:
            # Fallback to AST parsing
            try:
                tree = ast.parse(code)
                cyclomatic = self._estimate_ast_complexity(tree)
                funcs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                function_count = len(funcs)
                method = "ast"
            except Exception:
                # Syntax error or parser error fallback to heuristic
                return self._analyze_heuristic(code, lines, total_lines, blank_ratio, duplicate_ratio, "python")

        # Comment ratio
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        comment_ratio = comment_lines / total_lines

        # Docstring ratio for python via AST
        docstring_ratio = 0.0
        avg_func_len = 0.0
        try:
            tree = ast.parse(code)
            funcs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if funcs:
                docstring_count = sum(1 for f in funcs if ast.get_docstring(f) is not None)
                docstring_ratio = docstring_count / len(funcs)
                # Estimate average function length
                total_func_lines = 0
                for f in funcs:
                    # Estimate length from end_lineno (python 3.8+) or body length
                    end = getattr(f, "end_lineno", f.lineno + len(f.body))
                    total_func_lines += max(1, end - f.lineno)
                avg_func_len = total_func_lines / len(funcs)
        except Exception:
            pass

        return CodeQualityResult(
            cyclomatic_complexity=float(cyclomatic),
            function_count=function_count,
            avg_function_length=avg_func_len,
            comment_ratio=comment_ratio,
            docstring_ratio=docstring_ratio,
            duplicate_ratio=duplicate_ratio,
            total_lines=total_lines,
            blank_ratio=blank_ratio,
            language="python",
            analysis_method=method,
        )

    def _analyze_heuristic(
        self,
        code: str,
        lines: list[str],
        total_lines: int,
        blank_ratio: float,
        duplicate_ratio: float,
        language: str,
    ) -> CodeQualityResult:
        """Heuristic analysis for non-Python or syntax-invalid Python code."""
        # 1. Count functions via regex patterns
        func_patterns = {
            "python": r"^\s*(def|class)\s+\w+",
            "javascript": r"\b(function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>|\bclass\s+\w+)",
            "typescript": r"\b(function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>|\bclass\s+\w+)",
            "cpp": r"\b(void|int|double|float|char|bool|auto)\s+\w+\s*\([^)]*\)\s*\{",
            "c": r"\b(void|int|double|float|char|bool)\s+\w+\s*\([^)]*\)\s*\{",
            "java": r"\b(public|protected|private|static)\s+[\w<>]+\s+\w+\s*\([^)]*\)",
            "rust": r"\b(fn\s+\w+|struct\s+\w+|impl\s+\w+)",
            "go": r"\b(func\s+\w+|type\s+\w+\s+struct)",
        }
        pat = func_patterns.get(language, r"\b(def|fn|function|func)\b")
        function_count = sum(1 for line in lines if re.search(pat, line))

        # 2. Count comments
        comment_lines = 0
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.startswith(("//", "#", "--", "/*", "*", "'''", '"""')):
                comment_lines += 1
            elif "/*" in line and "*/" in line:
                comment_lines += 1

        comment_ratio = comment_lines / total_lines

        # 3. Simple cyclomatic complexity estimate (keyword based)
        # 1 base + number of branch keywords (if, else, for, while, switch, catch)
        branch_keywords = re.findall(r"\b(if|else|elif|for|while|case|catch|&&|\|\|)\b", code)
        cyclomatic = 1.0 + len(branch_keywords)

        # 4. Average function length estimate
        avg_func_len = 0.0
        if function_count > 0:
            avg_func_len = (total_lines - comment_lines - int(blank_ratio * total_lines)) / function_count

        # 5. Docstring ratio heuristic
        docstring_count = sum(1 for line in lines if "/**" in line or "///" in line or '"""' in line or "'''" in line)
        docstring_ratio = docstring_count / max(function_count, 1)

        return CodeQualityResult(
            cyclomatic_complexity=float(cyclomatic),
            function_count=function_count,
            avg_function_length=avg_func_len,
            comment_ratio=comment_ratio,
            docstring_ratio=min(1.0, docstring_ratio),
            duplicate_ratio=duplicate_ratio,
            total_lines=total_lines,
            blank_ratio=blank_ratio,
            language=language,
            analysis_method="heuristic",
        )

    @staticmethod
    def _estimate_ast_complexity(tree: ast.AST) -> float:
        """Estimate Python cyclomatic complexity via AST node counts."""
        complexity = 1.0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1.0
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
