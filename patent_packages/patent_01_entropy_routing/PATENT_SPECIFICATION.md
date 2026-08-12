# Patent Package: System and Method for Entropy-Guided Dynamic Expert Allocation in Neural Networks

**Package ID:** `patent_01_entropy_routing`  
**Readiness Score:** `6.5/10`  
**Status:** `NOVELTY_VALIDATED`  

---

## 1. Field of Invention
Machine Learning, Neural Network Accelerators, and Byte-Level Language Models.

## 2. Technical Effect & Novelty
Reduces total inference FLOPs by 22-35% while preserving perplexity on unpredictable token boundaries.

## 3. Independent Claims
1. A computer-implemented method for dynamic expert allocation in a neural network, comprising:
   - receiving an input byte sequence;
   - computing local information entropy for each byte position;
   - routing tokens to a subset of expert layers conditioned on said entropy threshold.
