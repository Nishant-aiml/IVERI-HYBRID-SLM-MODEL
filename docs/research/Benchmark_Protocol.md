# IVERI CORE — Benchmark Protocol

This document defines execution parameters for functional capabilities evaluations.

---

## 1. Coding Benchmarks (HumanEval & MBPP)

To prevent code injection hazards and measure accuracy:
- **Sandbox Isolation:** All code snippets must run in a separate sandboxed Python subprocess.
- **Timeouts:** A strict timeout of 5.0 seconds per task is enforced to catch infinite loops.
- **Pass@1 Metric:** Calculated by checking compile and runtime outputs against deterministic test cases.

---

## 2. Long-Context Benchmarks (Needle in a Haystack)

Evaluates retrieval capacity under context limits:
- **Needle insertion:** Inset target facts ("secret keys") at uniform position ratios ($10\%, 30\%, 50\%, 70\%, 90\%$) across sequence scales ($2\text{k}$ to $128\text{k}$).
- **Output Inspection:** Checks if the decoded response exactly matches the target fact characters.

---

## 3. Instruction & Safety Benchmarks

- **Prompt Suite:** Run evaluations on the 35 fixed CS/General QA prompts.
- **Safety Audits:** Alignment inspector scans generated text for over-refusals, mode collapse loops, and reward hacking signs.
