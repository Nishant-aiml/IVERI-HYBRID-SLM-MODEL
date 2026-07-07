# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Code quality inspector for IVERI CORE coding evaluation (Phase 3.3).

Mirrors :mod:`evaluation.response_inspector` but specialised for code responses:

- **Syntax validation** — Python via ``ast.parse``; other languages via
  ``tree_sitter`` if available, else ``None`` (not checked / not penalised).
- **Language detection** — reads ``### Language: <lang>`` from the first 100
  bytes of the response.
- **Code keyword detection** — checks for common programming keywords across
  many languages.
- **Entropy** — Shannon entropy of the byte frequency distribution.
- **Batch aggregation** — ``inspect_batch`` returns a compact stats dict.

Examples
--------
>>> from evaluation.code_inspector import CodeInspector
>>> insp = CodeInspector()
>>> result = insp.inspect_bytes(b"def hello():\\n    return 'world'\\n")
>>> result.syntax_valid
True
>>> result.language_detected
'python'
"""

from __future__ import annotations

import ast
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Optional tree-sitter ───────────────────────────────────────────────────

try:
    import tree_sitter  # type: ignore

    _TREE_SITTER_AVAILABLE = True
    logger.debug("tree_sitter available — non-Python syntax checking enabled.")
except ImportError:
    _TREE_SITTER_AVAILABLE = False
    logger.debug(
        "tree_sitter not available — non-Python syntax_valid will be None (not penalised)."
    )

# ── Constants ──────────────────────────────────────────────────────────────

_LANGUAGE_HEADER_RE = re.compile(
    rb"###\s*Language:\s*([A-Za-z0-9_+#\-]+)", re.IGNORECASE
)

_CODE_KEYWORDS: frozenset[str] = frozenset(
    [
        "def",
        "return",
        "for",
        "while",
        "class",
        "function",
        "if",
        "else",
        "import",
        "var",
        "let",
        "const",
        "int",
        "void",
        "fn",
        "pub",
    ]
)

_MIN_CODE_LENGTH: int = 1


# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass
class CodeInspectionResult:
    """Result of inspecting a single code response.

    Attributes
    ----------
    length:
        Total byte length of the response.
    is_valid:
        ``True`` if no disqualifying issues detected.
    syntax_valid:
        ``True``/``False`` for checked languages; ``None`` when syntax was
        **not checked** (non-Python without tree-sitter).  ``None`` is never
        penalised.
    has_code_keywords:
        ``True`` if at least one recognised programming keyword was found.
    has_code_block:
        ``True`` if the response contains a fenced code block (`` ``` ``).
    entropy:
        Shannon entropy (bits/byte) of the byte distribution.
    issues:
        List of detected issue labels.
    language_detected:
        Language string from the ``### Language:`` header, or ``""`` if absent.
    """

    length: int = 0
    is_valid: bool = True
    syntax_valid: bool | None = None
    has_code_keywords: bool = False
    has_code_block: bool = False
    entropy: float = 0.0
    issues: list[str] = field(default_factory=list)
    language_detected: str = ""


# ── Inspector ──────────────────────────────────────────────────────────────


class CodeInspector:
    """Inspect generated code responses for quality and correctness.

    Parameters
    ----------
    min_length:
        Minimum byte length for a non-empty response.
    """

    def __init__(self, min_length: int = _MIN_CODE_LENGTH) -> None:
        self.min_length = min_length

    # ── Public API ─────────────────────────────────────────────────────

    def inspect_bytes(self, response_bytes: bytes) -> CodeInspectionResult:
        """Inspect a raw byte code response.

        Parameters
        ----------
        response_bytes:
            Raw bytes (UTF-8 code response).

        Returns
        -------
        CodeInspectionResult
        """
        result = CodeInspectionResult()
        issues: list[str] = []

        result.length = len(response_bytes)

        # 1. Detect language from header
        result.language_detected = _detect_language(response_bytes)

        # 2. Decode
        try:
            code_text = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            code_text = response_bytes.decode("utf-8", errors="replace")
            issues.append("utf8_corruption")

        # 3. Empty check
        if result.length < self.min_length or not code_text.strip():
            issues.append("empty")
            result.issues = issues
            result.is_valid = False
            return result

        # 4. Entropy
        result.entropy = _byte_entropy(response_bytes)

        # 5. Code keyword detection
        result.has_code_keywords = _has_code_keywords(code_text)
        if not result.has_code_keywords:
            issues.append("no_code_keywords")

        # 6. Code block detection
        result.has_code_block = "```" in code_text

        # 7. Syntax validation
        lang = result.language_detected.lower()
        if lang == "python" or (not lang and _looks_like_python(code_text)):
            result.syntax_valid = _validate_python_syntax(code_text, issues)
        elif _TREE_SITTER_AVAILABLE and lang:
            result.syntax_valid = _validate_with_tree_sitter(
                code_text, lang, issues
            )
        else:
            # Non-Python without tree-sitter: do NOT penalise
            result.syntax_valid = None

        result.issues = issues
        result.is_valid = len(issues) == 0
        return result

    def score_response(self, response_bytes: bytes) -> float:
        """Compute a scalar quality score in ``[0.0, 1.0]``.

        Parameters
        ----------
        response_bytes:
            Raw byte code response.

        Returns
        -------
        float
            Quality score (higher = better).
        """
        result = self.inspect_bytes(response_bytes)

        if "empty" in result.issues:
            return 0.0

        penalty = 0.0

        # Penalise confirmed syntax errors (None = not checked, no penalty)
        if result.syntax_valid is False:
            penalty += 0.4

        # Penalise missing code keywords
        if not result.has_code_keywords:
            penalty += 0.2

        # Minor UTF-8 penalty
        if "utf8_corruption" in result.issues:
            penalty += 0.1

        # Entropy bonus (normalised; max = log2(256) = 8)
        entropy_score = min(result.entropy / 8.0, 1.0)

        base = max(0.0, 1.0 - penalty)
        return (base + entropy_score) / 2.0

    def inspect_batch(
        self, responses: list[bytes]
    ) -> dict[str, float | int | dict]:
        """Inspect a batch of code responses and return aggregate statistics.

        Parameters
        ----------
        responses:
            List of raw byte code responses.

        Returns
        -------
        dict
            Keys: ``valid_ratio``, ``avg_entropy``, ``syntax_valid_ratio``,
            ``has_code_keyword_ratio``, ``issue_counts``, ``avg_length``,
            ``empty_count``.

            ``syntax_valid_ratio`` = ``(#syntax_valid is True)`` /
            ``(#syntax_valid is not None)``.  Returns ``0.0`` when all are
            ``None``.
        """
        if not responses:
            return {
                "valid_ratio": 0.0,
                "avg_entropy": 0.0,
                "syntax_valid_ratio": 0.0,
                "has_code_keyword_ratio": 0.0,
                "issue_counts": {},
                "avg_length": 0.0,
                "empty_count": 0,
            }

        results = [self.inspect_bytes(r) for r in responses]
        n = len(results)

        valid_count = sum(1 for r in results if r.is_valid)
        empty_count = sum(1 for r in results if "empty" in r.issues)

        # syntax_valid_ratio only over responses where check was performed
        checked = [r for r in results if r.syntax_valid is not None]
        syntax_valid_ratio = (
            sum(1 for r in checked if r.syntax_valid) / len(checked)
            if checked
            else 0.0
        )

        keyword_count = sum(1 for r in results if r.has_code_keywords)

        issue_counter: Counter[str] = Counter()
        for r in results:
            for issue in r.issues:
                issue_counter[issue] += 1

        return {
            "valid_ratio": valid_count / n,
            "avg_entropy": sum(r.entropy for r in results) / n,
            "syntax_valid_ratio": syntax_valid_ratio,
            "has_code_keyword_ratio": keyword_count / n,
            "issue_counts": dict(issue_counter),
            "avg_length": sum(r.length for r in results) / n,
            "empty_count": empty_count,
        }


# ── Private helpers ────────────────────────────────────────────────────────


def _detect_language(response_bytes: bytes) -> str:
    """Extract language from ``### Language: <lang>`` in the first 100 bytes."""
    header = response_bytes[:100]
    match = _LANGUAGE_HEADER_RE.search(header)
    if match:
        return match.group(1).decode("ascii", errors="replace").lower()
    return ""


def _looks_like_python(code: str) -> bool:
    """Heuristic: does the code look like Python?"""
    stripped = code.strip()
    return bool(
        re.search(r"\bdef\s+\w+\s*\(", stripped)
        or re.search(r"\bclass\s+\w+", stripped)
        or re.search(r"\bimport\s+\w+", stripped)
        or stripped.startswith("#")
    )


def _validate_python_syntax(
    code: str, issues: list[str]
) -> bool:
    """Attempt ``ast.parse``; append ``syntax_error`` to *issues* on failure."""
    # Strip fenced code blocks if present
    clean = _strip_fenced_code(code)
    try:
        ast.parse(clean)
        return True
    except SyntaxError as exc:
        logger.debug("Python syntax error: %s", exc)
        issues.append("syntax_error")
        return False


def _validate_with_tree_sitter(
    code: str, language: str, issues: list[str]
) -> bool | None:
    """Attempt tree-sitter parse.  Returns ``None`` if language unsupported."""
    try:
        import tree_sitter_languages  # type: ignore

        lang_obj = tree_sitter_languages.get_language(language)
        parser = tree_sitter.Parser()  # type: ignore
        parser.set_language(lang_obj)
        tree = parser.parse(code.encode("utf-8", errors="replace"))
        if tree.root_node.has_error:
            issues.append("syntax_error")
            return False
        return True
    except Exception as exc:  # language not supported etc.
        logger.debug(
            "tree-sitter parse failed for language %r: %s — returning None",
            language,
            exc,
        )
        return None


def _strip_fenced_code(text: str) -> str:
    """Remove outermost fenced code block markers."""
    lines = text.splitlines()
    # Remove leading/trailing ``` fences
    start = 0
    end = len(lines)
    if lines and lines[0].startswith("```"):
        start = 1
    if lines and lines[-1].strip() == "```":
        end -= 1
    return "\n".join(lines[start:end])


def _has_code_keywords(code: str) -> bool:
    """Return ``True`` if any recognised code keyword is present."""
    words = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    return bool(words & _CODE_KEYWORDS)


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy (bits/byte) of *data*."""
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        entropy -= p * math.log2(p)
    return entropy
