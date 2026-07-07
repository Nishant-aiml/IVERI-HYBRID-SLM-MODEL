# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Prompt suite for Phase 3.4 Preference Optimization.

Contains exactly 50 deterministic inline engineering, scientific, and reasoning prompts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlignmentEvalPrompt:
    """Dataclass representing a preference evaluation prompt.

    Attributes
    ----------
    prompt_id:
        Unique string identifier.
    category:
        Reasoning/Coding/Systems domain.
    instruction:
        The instruction text.
    expected_keywords:
        Keywords expected in high-quality aligned outputs.
    reference_response:
        Ground truth reference solution.
    """

    prompt_id: str
    category: str
    instruction: str
    expected_keywords: list[str] = field(default_factory=list)
    reference_response: str = ""


class AlignmentPromptSuite:
    """Deterministic prompt suite of exactly 50 prompts across 11 key engineering categories."""

    def __init__(self) -> None:
        self._prompts: list[AlignmentEvalPrompt] = []
        self._initialize_prompts()

    def get_all(self) -> list[AlignmentEvalPrompt]:
        """Return all 50 evaluation prompts."""
        return list(self._prompts)

    def get_by_category(self, category: str) -> list[AlignmentEvalPrompt]:
        """Filter prompts by category."""
        return [p for p in self._prompts if p.category.lower() == category.lower()]

    def get_suite_hash(self) -> str:
        """Compute SHA-256 fingerprint of the prompt contents."""
        joined = "".join(p.instruction for p in self._prompts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _initialize_prompts(self) -> None:
        # Exposes 50 distinct prompts across 11 categories
        prompts_data = [
            # ── 1. General Reasoning (5 prompts) ──────────────────────────
            {
                "id": "reasoning_01",
                "cat": "general_reasoning",
                "inst": "If all roses are flowers and some flowers fade quickly, does it follow logically that some roses fade quickly? Explain.",
                "keywords": ["logical", "valid", "invalid", "syllogism"],
                "ref": "No. The premise 'some flowers fade quickly' does not guarantee that those fading flowers are roses."
            },
            {
                "id": "reasoning_02",
                "cat": "general_reasoning",
                "inst": "You have a 3-liter jug and a 5-liter jug. Explain how to measure exactly 4 liters of water.",
                "keywords": ["pour", "empty", "jug", "liters"],
                "ref": "Fill 5L jug. Pour into 3L jug, leaving 2L. Empty 3L jug. Pour 2L into 3L jug. Fill 5L jug. Pour into 3L jug until full (needs 1L). 5L jug now has exactly 4L."
            },
            {
                "id": "reasoning_03",
                "cat": "general_reasoning",
                "inst": "Explain the difference between inductive and deductive reasoning with examples.",
                "keywords": ["inductive", "deductive", "premises", "general", "specific"],
                "ref": "Deductive starts with general rules and guarantees specific conclusions. Inductive goes from specific observations to probable generalizations."
            },
            {
                "id": "reasoning_04",
                "cat": "general_reasoning",
                "inst": "A farmer has chickens and rabbits. There are 35 heads and 94 legs. How many chickens and rabbits does he have?",
                "keywords": ["heads", "legs", "rabbits", "chickens", "23", "12"],
                "ref": "23 chickens and 12 rabbits. Let c + r = 35 and 2c + 4r = 94. Solving yields c=23, r=12."
            },
            {
                "id": "reasoning_05",
                "cat": "general_reasoning",
                "inst": "What is the logical fallacy in the argument: 'Nobody has proven that ghosts don't exist, therefore they must exist'?",
                "keywords": ["fallacy", "ignorance", "ad ignorantiam", "burden of proof"],
                "ref": "Appeal to Ignorance (Argumentum ad Ignorantiam)."
            },

            # ── 2. Coding (5 prompts) ─────────────────────────────────────
            {
                "id": "coding_01",
                "cat": "coding",
                "inst": "Write a Python function to check if a string is a palindrome.",
                "keywords": ["def", "palindrome", "return", "::", "lower"],
                "ref": "def is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]"
            },
            {
                "id": "coding_02",
                "cat": "coding",
                "inst": "Write a Python function that merges two sorted lists into one sorted list.",
                "keywords": ["def", "merge", "while", "append", "sorted"],
                "ref": "def merge_sorted(l1, l2):\n    res = []\n    i, j = 0, 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] < l2[j]:\n            res.append(l1[i]); i += 1\n        else:\n            res.append(l2[j]); j += 1\n    res.extend(l1[i:]); res.extend(l2[j:])\n    return res"
            },
            {
                "id": "coding_03",
                "cat": "coding",
                "inst": "Write a Python generator function to yield the Fibonacci sequence up to N elements.",
                "keywords": ["yield", "generator", "def", "fibonacci"],
                "ref": "def fib_gen(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b"
            },
            {
                "id": "coding_04",
                "cat": "coding",
                "inst": "Write a Python class for a Node in a Doubly Linked List.",
                "keywords": ["class", "self", "next", "prev", "value"],
                "ref": "class Node:\n    def __init__(self, val=0):\n        self.val = val\n        self.next = None\n        self.prev = None"
            },
            {
                "id": "coding_05",
                "cat": "coding",
                "inst": "Write a Python script that reads a text file and counts the frequency of each word.",
                "keywords": ["open", "read", "split", "dict", "count"],
                "ref": "def word_count(filepath):\n    counts = {}\n    with open(filepath, 'r') as f:\n        for word in f.read().split():\n            counts[word] = counts.get(word, 0) + 1\n    return counts"
            },

            # ── 3. Algorithms (5 prompts) ─────────────────────────────────
            {
                "id": "algo_01",
                "cat": "algorithms",
                "inst": "Implement Binary Search in Python and state its time complexity.",
                "keywords": ["binary", "search", "mid", "O(log n)", "low", "high"],
                "ref": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"
            },
            {
                "id": "algo_02",
                "cat": "algorithms",
                "inst": "Implement Quicksort in Python using list comprehensions.",
                "keywords": ["quicksort", "pivot", "comprehension", "partition"],
                "ref": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
            },
            {
                "id": "algo_03",
                "cat": "algorithms",
                "inst": "Write a Python function to find the shortest path in a graph using Dijkstra's algorithm.",
                "keywords": ["dijkstra", "heapq", "priority", "queue", "distance"],
                "ref": "import heapq\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    queue = [(0, start)]\n    while queue:\n        dist, node = heapq.heappop(queue)\n        if dist > distances[node]: continue\n        for neighbor, weight in graph[node].items():\n            new_dist = dist + weight\n            if new_dist < distances[neighbor]:\n                distances[neighbor] = new_dist\n                heapq.heappush(queue, (new_dist, neighbor))\n    return distances"
            },
            {
                "id": "algo_04",
                "cat": "algorithms",
                "inst": "Explain the concept of Dynamic Programming and give an example of memoization vs tabulation.",
                "keywords": ["memoization", "tabulation", "dynamic programming", "subproblems", "overlapping"],
                "ref": "Dynamic programming solves complex problems by breaking them into overlapping subproblems. Memoization is top-down caching. Tabulation is bottom-up table filling."
            },
            {
                "id": "algo_05",
                "cat": "algorithms",
                "inst": "Write a Python function to detect a cycle in a directed graph using Depth First Search (DFS).",
                "keywords": ["dfs", "cycle", "visited", "stack", "recursion"],
                "ref": "def has_cycle(graph):\n    visited = set(); rec_stack = set()\n    def dfs(node):\n        visited.add(node); rec_stack.add(node)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                if dfs(neighbor): return True\n            elif neighbor in rec_stack: return True\n        rec_stack.remove(node); return False\n    return any(dfs(node) for node in graph if node not in visited)"
            },

            # ── 4. Debugging (5 prompts) ──────────────────────────────────
            {
                "id": "debug_01",
                "cat": "debugging",
                "inst": "Fix this code that raises an error when modifying a list during iteration:\n```python\nitems = [1, 2, 3]\nfor x in items:\n    if x == 2:\n        items.remove(x)\n```",
                "keywords": ["copy", "items[:]", "list comprehension", "remove"],
                "ref": "Modifying a list while iterating over it causes indexing issues. Fix:\n```python\nitems = [x for x in items if x != 2]\n```"
            },
            {
                "id": "debug_02",
                "cat": "debugging",
                "inst": "Identify and fix the bug in this binary search mid-calculation to prevent overflow in static-typed languages:\n`mid = (low + high) / 2`",
                "keywords": ["overflow", "low + (high - low) / 2", "mid"],
                "ref": "If low and high are large, low + high can overflow. Correct formula: `mid = low + (high - low) // 2`."
            },
            {
                "id": "debug_03",
                "cat": "debugging",
                "inst": "Why does this class raise an error, and how do you fix it?\n```python\nclass A:\n    def method(val):\n        print(val)\n```",
                "keywords": ["self", "method", "instance"],
                "ref": "Instance methods must accept 'self' as their first argument: `def method(self, val):`."
            },
            {
                "id": "debug_04",
                "cat": "debugging",
                "inst": "Fix this Python function which has a mutable default argument bug:\n```python\ndef add_to_list(val, my_list=[]):\n    my_list.append(val)\n    return my_list\n```",
                "keywords": ["None", "default", "mutable", "my_list is None"],
                "ref": "Default arguments are evaluated once at function definition. Fix:\n```python\ndef add_to_list(val, my_list=None):\n    if my_list is None:\n        my_list = []\n    my_list.append(val)\n    return my_list\n```"
            },
            {
                "id": "debug_05",
                "cat": "debugging",
                "inst": "Why does this comparison return False and how do you fix it?\n`x = 0.1 + 0.2; print(x == 0.3)`",
                "keywords": ["float", "precision", "math.isclose", "round"],
                "ref": "Floating-point precision limits cause 0.1 + 0.2 to evaluate to 0.30000000000000004. Fix: `math.isclose(0.1 + 0.2, 0.3)`."
            },

            # ── 5. DBMS (5 prompts) ────────────────────────────────────────
            {
                "id": "db_01",
                "cat": "dbms",
                "inst": "Explain the ACID properties of a database transaction.",
                "keywords": ["atomicity", "consistency", "isolation", "durability", "acid"],
                "ref": "Atomicity (all or nothing), Consistency (preserves rules), Isolation (independent concurrent execution), Durability (persists after commit)."
            },
            {
                "id": "db_02",
                "cat": "dbms",
                "inst": "Write a SQL query to find the second highest salary from an Employee table.",
                "keywords": ["select", "max", "employee", "offset", "limit", "where"],
                "ref": "SELECT MAX(Salary) FROM Employee WHERE Salary < (SELECT MAX(Salary) FROM Employee);"
            },
            {
                "id": "db_03",
                "cat": "dbms",
                "inst": "Explain the difference between Clustered and Non-Clustered Indexes in SQL databases.",
                "keywords": ["clustered", "non-clustered", "order", "leaf", "index"],
                "ref": "Clustered indexes define physical data storage order (1 per table). Non-clustered indexes maintain a separate key-pointer structure (multiple allowed)."
            },
            {
                "id": "db_04",
                "cat": "dbms",
                "inst": "What are SQL joins? Briefly explain INNER, LEFT, RIGHT, and FULL joins.",
                "keywords": ["inner", "left", "right", "full", "join"],
                "ref": "INNER: matching in both. LEFT: all left + matching right. RIGHT: all right + matching left. FULL: all rows when there is a match in either."
            },
            {
                "id": "db_05",
                "cat": "dbms",
                "inst": "Explain database normalization up to Third Normal Form (3NF).",
                "keywords": ["1nf", "2nf", "3nf", "atomic", "partial", "transitive"],
                "ref": "1NF: atomic values. 2NF: 1NF + no partial dependency. 3NF: 2NF + no transitive dependencies."
            },

            # ── 6. OS (5 prompts) ──────────────────────────────────────────
            {
                "id": "os_01",
                "cat": "os",
                "inst": "What is virtual memory and how does paging solve external memory fragmentation?",
                "keywords": ["virtual", "memory", "paging", "pages", "frames", "fragmentation"],
                "ref": "Virtual memory maps process addresses to physical frames. Paging breaks memory into fixed blocks, eliminating external fragmentation."
            },
            {
                "id": "os_02",
                "cat": "os",
                "inst": "Explain the difference between a Process and a Thread.",
                "keywords": ["process", "thread", "memory", "resource", "sharing", "address space"],
                "ref": "A process is an isolated executing program with its own memory. A thread is a lightweight execution unit sharing resources within a process."
            },
            {
                "id": "os_03",
                "cat": "os",
                "inst": "Explain the four necessary conditions for a Deadlock to occur.",
                "keywords": ["mutual exclusion", "hold and wait", "no preemption", "circular wait", "deadlock"],
                "ref": "Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait."
            },
            {
                "id": "os_04",
                "cat": "os",
                "inst": "What is a system call? Give three examples in Unix/Linux.",
                "keywords": ["system call", "kernel", "user mode", "fork", "read", "write"],
                "ref": "An interface between user application and OS kernel. Examples: fork(), read(), write(), exec()."
            },
            {
                "id": "os_05",
                "cat": "os",
                "inst": "Explain the difference between Preemptive and Non-Preemptive scheduling algorithms.",
                "keywords": ["preemptive", "non-preemptive", "scheduler", "context switch"],
                "ref": "Preemptive: scheduler can interrupt running processes (e.g. Round Robin). Non-preemptive: process runs until yield/block (e.g. FCFS)."
            },

            # ── 7. CN (5 prompts) ──────────────────────────────────────────
            {
                "id": "cn_01",
                "cat": "cn",
                "inst": "Explain the OSI 7-layer model and list the layers in order.",
                "keywords": ["physical", "datalink", "network", "transport", "session", "presentation", "application"],
                "ref": "Physical, Data Link, Network, Transport, Session, Presentation, Application."
            },
            {
                "id": "cn_02",
                "cat": "cn",
                "inst": "Explain the difference between TCP and UDP protocols.",
                "keywords": ["tcp", "udp", "connection-oriented", "reliable", "connectionless"],
                "ref": "TCP is connection-oriented, reliable, guarantees order. UDP is connectionless, fast, unreliable."
            },
            {
                "id": "cn_03",
                "cat": "cn",
                "inst": "How does DNS resolve a domain name to an IP address? Explain the steps.",
                "keywords": ["dns", "resolver", "root", "tld", "authoritative", "ip"],
                "ref": "Query resolver -> Root nameserver -> TLD nameserver -> Authoritative nameserver -> IP address returned and cached."
            },
            {
                "id": "cn_04",
                "cat": "cn",
                "inst": "What is the three-way handshake in TCP/IP connection establishment?",
                "keywords": ["syn", "syn-ack", "ack", "handshake"],
                "ref": "Client sends SYN. Server replies SYN-ACK. Client sends ACK. Connection is established."
            },
            {
                "id": "cn_05",
                "cat": "cn",
                "inst": "Explain how a Router differs from a Switch in computer networking.",
                "keywords": ["router", "switch", "ip", "mac", "layer 3", "layer 2"],
                "ref": "Switch operates at Layer 2 (MAC addresses) for local networks. Router operates at Layer 3 (IP addresses) between different networks."
            },

            # ── 8. ML (5 prompts) ──────────────────────────────────────────
            {
                "id": "ml_01",
                "cat": "ml",
                "inst": "What is the difference between L1 and L2 regularization and how do they impact weights?",
                "keywords": ["l1", "l2", "lasso", "ridge", "sparsity", "absolute", "squared"],
                "ref": "L1 (Lasso) adds absolute weight penalty and drives weights to zero (sparsity). L2 (Ridge) adds squared weight penalty and shrinks weights close to zero."
            },
            {
                "id": "ml_02",
                "cat": "ml",
                "inst": "Explain the bias-variance tradeoff in Machine Learning.",
                "keywords": ["bias", "variance", "overfitting", "underfitting", "tradeoff"],
                "ref": "Bias represents error from assumptions (underfitting). Variance represents sensitivity to data noise (overfitting). Minimizing total error requires balancing both."
            },
            {
                "id": "ml_03",
                "cat": "ml",
                "inst": "Explain the mathematical formulation and optimization objective of Support Vector Machines (SVM).",
                "keywords": ["svm", "margin", "hyperplane", "support vector", "kernel"],
                "ref": "SVM maximizes the margin width between two classes separated by a decision hyperplane: min ||w||^2 subject to classification correctness constraints."
            },
            {
                "id": "ml_04",
                "cat": "ml",
                "inst": "What is Gradient Descent? Explain Stochastic, Batch, and Mini-batch variations.",
                "keywords": ["gradient descent", "stochastic", "batch", "mini-batch", "learning rate"],
                "ref": "Iterative optimization to minimize loss. Batch updates on all data. Stochastic on single samples. Mini-batch on subset chunks."
            },
            {
                "id": "ml_05",
                "cat": "ml",
                "inst": "Explain the concept of Cross-Validation, particularly K-Fold Cross Validation.",
                "keywords": ["cross-validation", "k-fold", "validation", "overfitting"],
                "ref": "Splits data into K folds. Trains K times, each time using a different fold for validation and K-1 folds for training. Averages scores."
            },

            # ── 9. AI (5 prompts) ──────────────────────────────────────────
            {
                "id": "ai_01",
                "cat": "ai",
                "inst": "Explain the core components of the Transformer architecture attention mechanism (Q, K, V).",
                "keywords": ["attention", "query", "key", "value", "scaled dot-product", "softmax"],
                "ref": "Query (current token search vector), Key (target token representation vector), and Value (content vector). Output = softmax(QK^T / sqrt(d))V."
            },
            {
                "id": "ai_02",
                "cat": "ai",
                "inst": "What is the difference between Supervised and Unsupervised learning? Provide examples of both.",
                "keywords": ["supervised", "unsupervised", "labelled", "unlabelled", "classification", "clustering"],
                "ref": "Supervised trains on labelled data (e.g. regression, SVM). Unsupervised trains on unlabelled data to find patterns (e.g. K-Means, PCA)."
            },
            {
                "id": "ai_03",
                "cat": "ai",
                "inst": "What are Hallucinations in LLMs and what techniques help mitigate them?",
                "keywords": ["hallucination", "rag", "decoding", "temperature", "grounding"],
                "ref": "LLMs generating false statements confidently. Mitigated by RAG, temperature adjustment, constraint decoding, and human preference optimization."
            },
            {
                "id": "ai_04",
                "cat": "ai",
                "inst": "Explain the Reinforcement Learning from Human Feedback (RLHF) pipeline for LLMs.",
                "keywords": ["rlhf", "sft", "reward model", "ppo", "preference"],
                "ref": "Phase 1: Supervised Fine-Tuning. Phase 2: Train a Reward Model on comparison rankings. Phase 3: Optimize SFT policy using PPO against reward model."
            },
            {
                "id": "ai_05",
                "cat": "ai",
                "inst": "What is parameter-efficient fine-tuning (PEFT)? Explain LoRA.",
                "keywords": ["peft", "lora", "low-rank", "adapter", "parameter-efficient"],
                "ref": "Fine-tuning a fraction of parameters. LoRA decomposes weight updates into low-rank matrices A and B (h = Wx + BAx), freezing base model."
            },

            # ── 10. Mathematics (3 prompts) ───────────────────────────────
            {
                "id": "math_01",
                "cat": "mathematics",
                "inst": "State Bayes' Theorem and define each of its components.",
                "keywords": ["bayes", "posterior", "prior", "likelihood", "marginal", "p(a|b)"],
                "ref": "P(A|B) = P(B|A)P(A) / P(B). P(A|B) is posterior, P(B|A) likelihood, P(A) prior, P(B) evidence / marginal likelihood."
            },
            {
                "id": "math_02",
                "cat": "mathematics",
                "inst": "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
                "keywords": ["eigenvalue", "determinant", "characteristic", "3", "1"],
                "ref": "det(A - lambda*I) = (2-lambda)^2 - 1 = lambda^2 - 4*lambda + 3 = 0. Eigenvalues are lambda = 1 and lambda = 3."
            },
            {
                "id": "math_03",
                "cat": "mathematics",
                "inst": "What is the central limit theorem and why is it important in statistical inference?",
                "keywords": ["central limit theorem", "normal distribution", "sample mean", "variance"],
                "ref": "As sample size grows, the distribution of sample means approaches a normal distribution, regardless of the population distribution shape."
            },

            # ── 11. Indian Engineering (2 prompts) ────────────────────────
            {
                "id": "indian_eng_01",
                "cat": "indian_engineering",
                "inst": "Explain the concept of LL(1) parsing in Compiler Design. What do the first L, second L, and 1 mean?",
                "keywords": ["compiler", "ll(1)", "left-to-right", "leftmost", "lookahead", "grammar"],
                "ref": "First L: left-to-right input scan. Second L: leftmost derivation. 1: one token of lookahead. Used in top-down parser design."
            },
            {
                "id": "indian_eng_02",
                "cat": "indian_engineering",
                "inst": "Explain Carnot Cycle and list the four thermodynamic processes in it.",
                "keywords": ["thermodynamics", "carnot", "isothermal expansion", "adiabatic expansion", "isothermal compression", "adiabatic compression"],
                "ref": "An idealized thermodynamic cycle. Processes: 1. Isothermal Expansion, 2. Adiabatic Expansion, 3. Isothermal Compression, 4. Adiabatic Compression."
            }
        ]

        for p in prompts_data:
            self._prompts.append(
                AlignmentEvalPrompt(
                    prompt_id=p["id"],
                    category=p["cat"],
                    instruction=p["inst"],
                    expected_keywords=p["keywords"],
                    reference_response=p["ref"]
                )
            )

        assert len(self._prompts) == 50, f"Prompt suite must contain exactly 50 prompts, got {len(self._prompts)}"
