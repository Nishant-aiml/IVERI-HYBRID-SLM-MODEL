# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Fixed 17-prompt coding evaluation suite for IVERI CORE (Phase 3.3).

Provides :class:`CodingPromptSuite` — a deterministic, versioned set of coding
prompts spread across six categories and three difficulty levels.  The prompts
are defined inline (no file I/O) so the suite hash is fully reproducible.

Examples
--------
>>> from evaluation.coding_prompt_suite import CodingPromptSuite
>>> suite = CodingPromptSuite()
>>> len(suite.get_all())
17
>>> suite.version
'3A-v1.0'
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Dataclass ──────────────────────────────────────────────────────────────


@dataclass
class CodeEvalPrompt:
    """A single coding evaluation prompt.

    Attributes
    ----------
    prompt_id:
        Unique identifier (e.g. ``"gen_001"``).
    category:
        One of ``"generation"``, ``"completion"``, ``"bug_fix"``,
        ``"explanation"``, ``"algorithm"``, ``"competitive"``.
    difficulty:
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    instruction:
        The text instruction given to the model.
    context:
        Optional additional context / code snippet.
    expected_keywords:
        Words that a correct solution is expected to contain.
    expected_language:
        Target programming language (default: ``"python"``).
    reference_solution:
        A reference solution string used for contamination checking.
    """

    prompt_id: str
    category: str
    difficulty: str
    instruction: str
    context: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    expected_language: str = "python"
    reference_solution: str = ""


# ── Suite ──────────────────────────────────────────────────────────────────


class CodingPromptSuite:
    """Deterministic 17-prompt coding evaluation suite.

    Category breakdown
    ------------------
    * generation  : 4 prompts
    * completion  : 3 prompts
    * bug_fix     : 3 prompts
    * explanation : 2 prompts
    * algorithm   : 3 prompts
    * competitive : 2 prompts

    Attributes
    ----------
    version:
        Suite version string — bump when prompts change.
    """

    version: str = "3A-v1.0"

    def __init__(self) -> None:
        self._prompts: list[CodeEvalPrompt] = _build_prompts()

    # ── Public API ─────────────────────────────────────────────────────

    def get_all(self) -> list[CodeEvalPrompt]:
        """Return all 17 prompts."""
        return list(self._prompts)

    def get_by_category(self, category: str) -> list[CodeEvalPrompt]:
        """Return prompts filtered by *category*.

        Parameters
        ----------
        category:
            Category name (case-insensitive).
        """
        return [p for p in self._prompts if p.category == category.lower()]

    def get_by_difficulty(self, difficulty: str) -> list[CodeEvalPrompt]:
        """Return prompts filtered by *difficulty*.

        Parameters
        ----------
        difficulty:
            Difficulty level (case-insensitive): ``"easy"``, ``"medium"``,
            or ``"hard"``.
        """
        return [p for p in self._prompts if p.difficulty == difficulty.lower()]

    def get_suite_hash(self) -> str:
        """Compute a SHA-256 fingerprint of the full prompt suite.

        The hash covers all ``instruction`` strings joined with ``\\n---\\n``
        so any prompt change is detectable.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest.
        """
        combined = "\n---\n".join(p.instruction for p in self._prompts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ── Prompt definitions ─────────────────────────────────────────────────────


def _build_prompts() -> list[CodeEvalPrompt]:
    """Construct the canonical 17-prompt list.  Inline, no file I/O."""

    prompts: list[CodeEvalPrompt] = []

    # ── GENERATION (4) ────────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="gen_001",
            category="generation",
            difficulty="easy",
            instruction=(
                "Write a Python function that takes a string and returns it reversed."
            ),
            expected_keywords=["def", "return", "[::-1]"],
            expected_language="python",
            reference_solution=(
                "def reverse_string(s: str) -> str:\n"
                "    return s[::-1]\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="gen_002",
            category="generation",
            difficulty="medium",
            instruction=(
                "Write a Python function to implement binary search on a sorted list."
                " The function should return the index of the target element,"
                " or -1 if not found."
            ),
            expected_keywords=["def", "return", "while", "mid", "-1"],
            expected_language="python",
            reference_solution=(
                "def binary_search(arr: list, target) -> int:\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="gen_003",
            category="generation",
            difficulty="easy",
            instruction=(
                "Write a Python function that checks if a given number is prime."
                " Return True if the number is prime, False otherwise."
            ),
            expected_keywords=["def", "return", "for", "if", "True", "False"],
            expected_language="python",
            reference_solution=(
                "def is_prime(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="gen_004",
            category="generation",
            difficulty="medium",
            instruction=(
                "Write a Python class implementing a stack data structure"
                " with push, pop, and peek methods."
                " Raise an IndexError when popping or peeking an empty stack."
            ),
            expected_keywords=["class", "def", "push", "pop", "peek", "IndexError"],
            expected_language="python",
            reference_solution=(
                "class Stack:\n"
                "    def __init__(self):\n"
                "        self._items = []\n"
                "\n"
                "    def push(self, item):\n"
                "        self._items.append(item)\n"
                "\n"
                "    def pop(self):\n"
                "        if not self._items:\n"
                "            raise IndexError('pop from empty stack')\n"
                "        return self._items.pop()\n"
                "\n"
                "    def peek(self):\n"
                "        if not self._items:\n"
                "            raise IndexError('peek from empty stack')\n"
                "        return self._items[-1]\n"
            ),
        )
    )

    # ── COMPLETION (3) ─────────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="cmp_001",
            category="completion",
            difficulty="easy",
            instruction=(
                "Complete the following Python function that computes the nth"
                " Fibonacci number:\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    # complete this\n"
                "```"
            ),
            expected_keywords=["return", "fibonacci", "n - 1", "n - 2"],
            expected_language="python",
            reference_solution=(
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="cmp_002",
            category="completion",
            difficulty="medium",
            instruction=(
                "Complete the following Python function that merges two sorted lists"
                " into a single sorted list:\n"
                "```python\n"
                "def merge_sorted(a: list, b: list) -> list:\n"
                "    result = []\n"
                "    i, j = 0, 0\n"
                "    # complete this\n"
                "```"
            ),
            expected_keywords=["while", "append", "return", "result"],
            expected_language="python",
            reference_solution=(
                "def merge_sorted(a: list, b: list) -> list:\n"
                "    result = []\n"
                "    i, j = 0, 0\n"
                "    while i < len(a) and j < len(b):\n"
                "        if a[i] <= b[j]:\n"
                "            result.append(a[i])\n"
                "            i += 1\n"
                "        else:\n"
                "            result.append(b[j])\n"
                "            j += 1\n"
                "    result.extend(a[i:])\n"
                "    result.extend(b[j:])\n"
                "    return result\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="cmp_003",
            category="completion",
            difficulty="hard",
            instruction=(
                "Complete the following Python generator that yields all permutations"
                " of a list:\n"
                "```python\n"
                "def permutations(lst: list):\n"
                "    if len(lst) <= 1:\n"
                "        yield lst\n"
                "        return\n"
                "    # complete this\n"
                "```"
            ),
            expected_keywords=["yield", "for", "permutations"],
            expected_language="python",
            reference_solution=(
                "def permutations(lst: list):\n"
                "    if len(lst) <= 1:\n"
                "        yield lst\n"
                "        return\n"
                "    for i, item in enumerate(lst):\n"
                "        rest = lst[:i] + lst[i + 1:]\n"
                "        for perm in permutations(rest):\n"
                "            yield [item] + perm\n"
            ),
        )
    )

    # ── BUG_FIX (3) ────────────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="fix_001",
            category="bug_fix",
            difficulty="easy",
            instruction=(
                "Fix the bug in this Python function:\n"
                "```python\n"
                "def find_max(lst):\n"
                "    max_val = lst[0]\n"
                "    for i in lst:\n"
                "        if i > max_val\n"
                "            max_val = i\n"
                "    return max_val\n"
                "```"
            ),
            expected_keywords=["def", "for", "if", "return", ":"],
            expected_language="python",
            reference_solution=(
                "def find_max(lst):\n"
                "    max_val = lst[0]\n"
                "    for i in lst:\n"
                "        if i > max_val:\n"
                "            max_val = i\n"
                "    return max_val\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="fix_002",
            category="bug_fix",
            difficulty="medium",
            instruction=(
                "Fix the off-by-one error in this Python function that removes"
                " duplicates from a sorted list:\n"
                "```python\n"
                "def remove_duplicates(lst):\n"
                "    result = [lst[0]]\n"
                "    for i in range(1, len(lst) + 1):  # bug here\n"
                "        if lst[i] != lst[i - 1]:\n"
                "            result.append(lst[i])\n"
                "    return result\n"
                "```"
            ),
            expected_keywords=["def", "for", "range", "return", "append"],
            expected_language="python",
            reference_solution=(
                "def remove_duplicates(lst):\n"
                "    result = [lst[0]]\n"
                "    for i in range(1, len(lst)):\n"
                "        if lst[i] != lst[i - 1]:\n"
                "            result.append(lst[i])\n"
                "    return result\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="fix_003",
            category="bug_fix",
            difficulty="hard",
            instruction=(
                "Fix the infinite recursion bug in this Python function that"
                " flattens a nested list:\n"
                "```python\n"
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(flatten(lst))  # bug here\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result\n"
                "```"
            ),
            expected_keywords=["def", "for", "isinstance", "extend", "flatten"],
            expected_language="python",
            reference_solution=(
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(flatten(item))\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result\n"
            ),
        )
    )

    # ── EXPLANATION (2) ───────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="exp_001",
            category="explanation",
            difficulty="easy",
            instruction=(
                "Explain what the following Python code does and identify its"
                " time complexity:\n"
                "```python\n"
                "def mystery(n):\n"
                "    count = 0\n"
                "    i = 1\n"
                "    while i <= n:\n"
                "        count += 1\n"
                "        i *= 2\n"
                "    return count\n"
                "```"
            ),
            expected_keywords=["logarithm", "O(log n)", "powers", "2", "count"],
            expected_language="python",
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="exp_002",
            category="explanation",
            difficulty="medium",
            instruction=(
                "Explain why the following Python code can raise a RuntimeError"
                " and how to fix it:\n"
                "```python\n"
                "d = {'a': 1, 'b': 2, 'c': 3}\n"
                "for key in d:\n"
                "    if d[key] == 2:\n"
                "        del d[key]\n"
                "```"
            ),
            expected_keywords=["RuntimeError", "dictionary", "iteration", "list", "copy"],
            expected_language="python",
        )
    )

    # ── ALGORITHM (3) ─────────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="alg_001",
            category="algorithm",
            difficulty="medium",
            instruction="Implement the quicksort algorithm in Python.",
            expected_keywords=["def", "quicksort", "pivot", "return", "for"],
            expected_language="python",
            reference_solution=(
                "def quicksort(arr: list) -> list:\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="alg_002",
            category="algorithm",
            difficulty="medium",
            instruction=(
                "Implement Dijkstra's shortest-path algorithm in Python."
                " The graph should be represented as an adjacency dictionary"
                " mapping node -> list of (neighbour, weight) pairs."
                " Return the distances dict from the start node."
            ),
            expected_keywords=["def", "heapq", "dist", "return", "for", "while"],
            expected_language="python",
            reference_solution=(
                "import heapq\n\n"
                "def dijkstra(graph: dict, start) -> dict:\n"
                "    dist = {start: 0}\n"
                "    heap = [(0, start)]\n"
                "    while heap:\n"
                "        d, u = heapq.heappop(heap)\n"
                "        if d > dist.get(u, float('inf')):\n"
                "            continue\n"
                "        for v, w in graph.get(u, []):\n"
                "            nd = d + w\n"
                "            if nd < dist.get(v, float('inf')):\n"
                "                dist[v] = nd\n"
                "                heapq.heappush(heap, (nd, v))\n"
                "    return dist\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="alg_003",
            category="algorithm",
            difficulty="hard",
            instruction=(
                "Implement a Least Recently Used (LRU) cache in Python with"
                " O(1) get and put operations."
                " The cache should have a fixed capacity; evict the least recently"
                " used item when full."
            ),
            expected_keywords=["class", "def", "get", "put", "OrderedDict", "capacity"],
            expected_language="python",
            reference_solution=(
                "from collections import OrderedDict\n\n"
                "class LRUCache:\n"
                "    def __init__(self, capacity: int):\n"
                "        self.capacity = capacity\n"
                "        self.cache = OrderedDict()\n\n"
                "    def get(self, key: int) -> int:\n"
                "        if key not in self.cache:\n"
                "            return -1\n"
                "        self.cache.move_to_end(key)\n"
                "        return self.cache[key]\n\n"
                "    def put(self, key: int, value: int) -> None:\n"
                "        if key in self.cache:\n"
                "            self.cache.move_to_end(key)\n"
                "        self.cache[key] = value\n"
                "        if len(self.cache) > self.capacity:\n"
                "            self.cache.popitem(last=False)\n"
            ),
        )
    )

    # ── COMPETITIVE (2) ───────────────────────────────────────────────

    prompts.append(
        CodeEvalPrompt(
            prompt_id="cpt_001",
            category="competitive",
            difficulty="medium",
            instruction=(
                "Given a positive integer N, find all prime numbers up to N"
                " using the Sieve of Eratosthenes algorithm."
                " Return the list of primes."
            ),
            expected_keywords=["def", "sieve", "for", "return", "primes", "range"],
            expected_language="python",
            reference_solution=(
                "def sieve_of_eratosthenes(n: int) -> list[int]:\n"
                "    if n < 2:\n"
                "        return []\n"
                "    is_prime = [True] * (n + 1)\n"
                "    is_prime[0] = is_prime[1] = False\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if is_prime[i]:\n"
                "            for j in range(i * i, n + 1, i):\n"
                "                is_prime[j] = False\n"
                "    return [i for i in range(n + 1) if is_prime[i]]\n"
            ),
        )
    )

    prompts.append(
        CodeEvalPrompt(
            prompt_id="cpt_002",
            category="competitive",
            difficulty="hard",
            instruction=(
                "Given a list of integers, find the maximum sum of a contiguous"
                " subarray (Kadane's algorithm). Return the maximum sum and the"
                " start and end indices of the subarray."
            ),
            expected_keywords=["def", "max", "return", "for", "current", "best"],
            expected_language="python",
            reference_solution=(
                "def max_subarray(nums: list[int]) -> tuple[int, int, int]:\n"
                "    best_sum = nums[0]\n"
                "    current_sum = nums[0]\n"
                "    best_start = best_end = start = 0\n"
                "    for i in range(1, len(nums)):\n"
                "        if current_sum + nums[i] < nums[i]:\n"
                "            current_sum = nums[i]\n"
                "            start = i\n"
                "        else:\n"
                "            current_sum += nums[i]\n"
                "        if current_sum > best_sum:\n"
                "            best_sum = current_sum\n"
                "            best_start = start\n"
                "            best_end = i\n"
                "    return best_sum, best_start, best_end\n"
            ),
        )
    )

    logger.debug(
        "CodingPromptSuite built: %d prompts across %d categories.",
        len(prompts),
        len({p.category for p in prompts}),
    )
    return prompts
