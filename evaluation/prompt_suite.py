# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Prompt suite for deterministic SFT evaluation of IVERI CORE.

Provides a fixed, version-stamped set of evaluation prompts across 14 categories.
Every evaluation run uses identical prompts for deterministic comparison across
checkpoints and training runs.

Categories
----------
1.  General Knowledge
2.  Reasoning
3.  Coding
4.  Debugging
5.  Python
6.  Algorithms
7.  DBMS
8.  Operating Systems
9.  Computer Networks
10. Machine Learning
11. Artificial Intelligence
12. Mathematics
13. Indian Engineering / GATE
14. Placement Interview

Examples
--------
>>> from evaluation.prompt_suite import PromptSuite
>>> suite = PromptSuite()
>>> prompts = suite.get_all()
>>> len(prompts) >= 14
True
>>> suite.get_category("coding")
[...]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Suite version ──────────────────────────────────────────────────────────

PROMPT_SUITE_VERSION: str = "1.0.0"
"""Version stamp for the prompt suite.  Increment when prompts change."""

# ── Prompt entry ───────────────────────────────────────────────────────────


@dataclass
class EvalPrompt:
    """A single evaluation prompt.

    Attributes
    ----------
    prompt_id:
        Unique identifier (e.g. ``"general_001"``).
    category:
        Category name (lowercase, underscores).
    instruction:
        The instruction text sent to the model.
    context:
        Optional additional context / input.
    expected_keywords:
        Keywords that a correct response should contain (used for scoring).
    difficulty:
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    """

    prompt_id: str
    category: str
    instruction: str
    context: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    difficulty: str = "medium"


# ── Canonical prompt set ───────────────────────────────────────────────────

_CANONICAL_PROMPTS: list[dict[str, Any]] = [
    # ── General Knowledge ──────────────────────────────────────────────
    {
        "prompt_id": "general_001",
        "category": "general_knowledge",
        "instruction": "What is the capital of France?",
        "expected_keywords": ["Paris"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "general_002",
        "category": "general_knowledge",
        "instruction": "Explain the greenhouse effect in simple terms.",
        "expected_keywords": ["carbon", "atmosphere", "temperature", "heat"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "general_003",
        "category": "general_knowledge",
        "instruction": "Who invented the World Wide Web and in what year?",
        "expected_keywords": ["Tim Berners-Lee", "1989", "1991"],
        "difficulty": "easy",
    },
    # ── Reasoning ─────────────────────────────────────────────────────
    {
        "prompt_id": "reasoning_001",
        "category": "reasoning",
        "instruction": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "expected_keywords": ["no", "cannot", "necessarily"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "reasoning_002",
        "category": "reasoning",
        "instruction": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "expected_keywords": ["5", "0.05", "five cents"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "reasoning_003",
        "category": "reasoning",
        "instruction": "What is the next number in the sequence: 2, 4, 8, 16, 32?",
        "expected_keywords": ["64"],
        "difficulty": "easy",
    },
    # ── Coding ────────────────────────────────────────────────────────
    {
        "prompt_id": "coding_001",
        "category": "coding",
        "instruction": "Write a Python function to check if a given number is prime.",
        "expected_keywords": ["def", "return", "prime", "for", "range"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "coding_002",
        "category": "coding",
        "instruction": "Implement a stack using a Python list with push, pop, and peek operations.",
        "expected_keywords": ["class", "append", "pop", "stack"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "coding_003",
        "category": "coding",
        "instruction": "Write a function that reverses a string without using slicing.",
        "expected_keywords": ["def", "return", "for", "reverse"],
        "difficulty": "easy",
    },
    # ── Debugging ─────────────────────────────────────────────────────
    {
        "prompt_id": "debugging_001",
        "category": "debugging",
        "instruction": "The following Python code raises an IndexError. Find and fix the bug:\n\ndef get_last(lst):\n    return lst[len(lst)]",
        "expected_keywords": ["lst[len(lst) - 1]", "lst[-1]", "index", "off by one"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "debugging_002",
        "category": "debugging",
        "instruction": "Explain what causes a RecursionError in Python and how to fix it.",
        "expected_keywords": ["base case", "recursion", "limit", "stack"],
        "difficulty": "medium",
    },
    # ── Python ────────────────────────────────────────────────────────
    {
        "prompt_id": "python_001",
        "category": "python",
        "instruction": "What is the difference between a list and a tuple in Python?",
        "expected_keywords": ["mutable", "immutable", "list", "tuple"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "python_002",
        "category": "python",
        "instruction": "Explain Python's GIL (Global Interpreter Lock) and its impact on multithreading.",
        "expected_keywords": ["GIL", "thread", "CPython", "multiprocessing"],
        "difficulty": "hard",
    },
    {
        "prompt_id": "python_003",
        "category": "python",
        "instruction": "What are Python decorators and how do they work? Give an example.",
        "expected_keywords": ["decorator", "@", "wrapper", "function"],
        "difficulty": "medium",
    },
    # ── Algorithms ────────────────────────────────────────────────────
    {
        "prompt_id": "algorithms_001",
        "category": "algorithms",
        "instruction": "Explain the time complexity of QuickSort in the best, average, and worst cases.",
        "expected_keywords": ["O(n log n)", "O(n^2)", "pivot"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "algorithms_002",
        "category": "algorithms",
        "instruction": "What is dynamic programming? Explain with the example of the Fibonacci sequence.",
        "expected_keywords": ["memoization", "subproblem", "Fibonacci", "overlapping"],
        "difficulty": "medium",
    },
    # ── DBMS ──────────────────────────────────────────────────────────
    {
        "prompt_id": "dbms_001",
        "category": "dbms",
        "instruction": "What is the difference between INNER JOIN and LEFT JOIN in SQL?",
        "expected_keywords": ["matching", "all rows", "NULL", "JOIN"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "dbms_002",
        "category": "dbms",
        "instruction": "Explain database normalization and its forms (1NF, 2NF, 3NF).",
        "expected_keywords": ["1NF", "2NF", "3NF", "redundancy", "dependency"],
        "difficulty": "medium",
    },
    # ── Operating Systems ─────────────────────────────────────────────
    {
        "prompt_id": "os_001",
        "category": "operating_systems",
        "instruction": "What is the difference between a process and a thread?",
        "expected_keywords": ["process", "thread", "memory", "lightweight"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "os_002",
        "category": "operating_systems",
        "instruction": "Explain deadlock and the four necessary conditions for it to occur.",
        "expected_keywords": ["mutual exclusion", "hold and wait", "no preemption", "circular wait"],
        "difficulty": "medium",
    },
    # ── Computer Networks ─────────────────────────────────────────────
    {
        "prompt_id": "networks_001",
        "category": "computer_networks",
        "instruction": "What is the difference between TCP and UDP? When would you use each?",
        "expected_keywords": ["reliable", "connectionless", "TCP", "UDP", "stream"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "networks_002",
        "category": "computer_networks",
        "instruction": "Explain how DNS (Domain Name System) works.",
        "expected_keywords": ["DNS", "resolver", "IP", "domain", "lookup"],
        "difficulty": "medium",
    },
    # ── Machine Learning ──────────────────────────────────────────────
    {
        "prompt_id": "ml_001",
        "category": "machine_learning",
        "instruction": "What is overfitting in machine learning and how can it be prevented?",
        "expected_keywords": ["overfitting", "regularization", "dropout", "validation"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "ml_002",
        "category": "machine_learning",
        "instruction": "Explain the difference between supervised, unsupervised, and reinforcement learning.",
        "expected_keywords": ["labeled", "unlabeled", "reward", "agent", "cluster"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "ml_003",
        "category": "machine_learning",
        "instruction": "What is gradient descent and how does it work in neural network training?",
        "expected_keywords": ["gradient", "loss", "weight", "learning rate", "backpropagation"],
        "difficulty": "medium",
    },
    # ── Artificial Intelligence ───────────────────────────────────────
    {
        "prompt_id": "ai_001",
        "category": "artificial_intelligence",
        "instruction": "What is the Transformer architecture and why did it revolutionize NLP?",
        "expected_keywords": ["attention", "Transformer", "BERT", "GPT", "parallel"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "ai_002",
        "category": "artificial_intelligence",
        "instruction": "Explain the concept of attention mechanism in neural networks.",
        "expected_keywords": ["attention", "query", "key", "value", "weight"],
        "difficulty": "hard",
    },
    # ── Mathematics ───────────────────────────────────────────────────
    {
        "prompt_id": "math_001",
        "category": "mathematics",
        "instruction": "What is Bayes' theorem? State the formula and give a real-world example.",
        "expected_keywords": ["P(A|B)", "posterior", "prior", "likelihood"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "math_002",
        "category": "mathematics",
        "instruction": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 7.",
        "expected_keywords": ["3x^2", "4x", "-5", "derivative"],
        "difficulty": "easy",
    },
    # ── Indian Engineering / GATE ─────────────────────────────────────
    {
        "prompt_id": "gate_001",
        "category": "indian_engineering_gate",
        "instruction": "In the context of GATE Computer Science, explain the concept of context-free grammars.",
        "expected_keywords": ["CFG", "production", "non-terminal", "terminal", "grammar"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "gate_002",
        "category": "indian_engineering_gate",
        "instruction": "What is the Pumping Lemma and how is it used to prove that a language is not regular?",
        "expected_keywords": ["pumping lemma", "regular", "contradiction", "string"],
        "difficulty": "hard",
    },
    {
        "prompt_id": "gate_003",
        "category": "indian_engineering_gate",
        "instruction": "Explain the dining philosophers problem and a solution using semaphores.",
        "expected_keywords": ["philosopher", "fork", "semaphore", "deadlock", "mutex"],
        "difficulty": "hard",
    },
    # ── Placement Interview ───────────────────────────────────────────
    {
        "prompt_id": "placement_001",
        "category": "placement_interview",
        "instruction": "Tell me about yourself in the context of a software engineering job interview.",
        "expected_keywords": ["experience", "skills", "project", "engineering"],
        "difficulty": "easy",
    },
    {
        "prompt_id": "placement_002",
        "category": "placement_interview",
        "instruction": "What is the difference between object-oriented programming and functional programming?",
        "expected_keywords": ["OOP", "functional", "class", "immutable", "side effects"],
        "difficulty": "medium",
    },
    {
        "prompt_id": "placement_003",
        "category": "placement_interview",
        "instruction": "Explain SOLID principles in software engineering.",
        "expected_keywords": ["Single", "Open", "Liskov", "Interface", "Dependency"],
        "difficulty": "medium",
    },
]


# ── Main suite class ───────────────────────────────────────────────────────


class PromptSuite:
    """Fixed, version-stamped prompt suite for deterministic SFT evaluation.

    All evaluation runs use identical prompts.  Prompts are ordered
    deterministically by ``prompt_id`` to ensure reproducibility.

    Parameters
    ----------
    prompts:
        Override the canonical prompt set.  Defaults to the built-in list.
    version:
        Suite version string.  Increment when prompts are modified.
    """

    def __init__(
        self,
        prompts: list[dict[str, Any]] | None = None,
        version: str = PROMPT_SUITE_VERSION,
    ) -> None:
        self.version = version
        raw = prompts if prompts is not None else _CANONICAL_PROMPTS
        # Build sorted, deterministic list
        self._prompts: list[EvalPrompt] = [
            EvalPrompt(**p)
            for p in sorted(raw, key=lambda x: x["prompt_id"])
        ]
        self._by_category: dict[str, list[EvalPrompt]] = {}
        for p in self._prompts:
            self._by_category.setdefault(p.category, []).append(p)

    # ── Public API ─────────────────────────────────────────────────────

    def get_all(self) -> list[EvalPrompt]:
        """Return all prompts (sorted by prompt_id).

        Returns
        -------
        list[EvalPrompt]
        """
        return list(self._prompts)

    def get_category(self, category: str) -> list[EvalPrompt]:
        """Return prompts for a specific category.

        Parameters
        ----------
        category:
            Category name (case-insensitive).

        Returns
        -------
        list[EvalPrompt]
            Matching prompts (empty list if category not found).
        """
        return self._by_category.get(category.lower(), [])

    def get_categories(self) -> list[str]:
        """Return sorted list of all category names.

        Returns
        -------
        list[str]
        """
        return sorted(self._by_category.keys())

    def get_by_difficulty(self, difficulty: str) -> list[EvalPrompt]:
        """Return all prompts of a given difficulty level.

        Parameters
        ----------
        difficulty:
            One of ``"easy"``, ``"medium"``, ``"hard"``.

        Returns
        -------
        list[EvalPrompt]
        """
        return [p for p in self._prompts if p.difficulty == difficulty]

    def as_instruction_list(self) -> list[str]:
        """Return all instruction texts in order.

        Returns
        -------
        list[str]
        """
        return [p.instruction for p in self._prompts]

    def as_dataset_dicts(self) -> list[dict[str, Any]]:
        """Return prompts formatted as Alpaca-style dicts (no output).

        Returns
        -------
        list[dict[str, Any]]
        """
        return [
            {
                "instruction": p.instruction,
                "input": p.context,
                "output": "",
            }
            for p in self._prompts
        ]

    def get_suite_hash(self) -> str:
        """Compute a SHA-256 hash of all prompt instructions (for version tracking).

        Returns
        -------
        str
            Hex digest.
        """
        content = json.dumps(
            [p.instruction for p in self._prompts],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        """Serialize the prompt suite to a JSON file.

        Parameters
        ----------
        path:
            Output file path.
        """
        out = {
            "version": self.version,
            "suite_hash": self.get_suite_hash(),
            "num_prompts": len(self._prompts),
            "categories": self.get_categories(),
            "prompts": [vars(p) for p in self._prompts],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PromptSuite:
        """Load a prompt suite from a JSON file.

        Parameters
        ----------
        path:
            Path to a previously saved prompt suite file.

        Returns
        -------
        PromptSuite
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(prompts=data["prompts"], version=data.get("version", PROMPT_SUITE_VERSION))

    def __len__(self) -> int:
        return len(self._prompts)

    def __repr__(self) -> str:
        return (
            f"PromptSuite(version={self.version!r}, "
            f"num_prompts={len(self._prompts)}, "
            f"categories={len(self._by_category)})"
        )
