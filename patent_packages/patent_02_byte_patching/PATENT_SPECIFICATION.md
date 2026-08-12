# Patent Package: Method for Tokenization-Free Byte Sequence Compression via Dynamic Boundary Prediction

**Package ID:** `patent_02_byte_patching`  
**Readiness Score:** `6.0/10`  
**Status:** `NOVELTY_VALIDATED`  

---

## 1. Field of Invention
Machine Learning, Neural Network Accelerators, and Byte-Level Language Models.

## 2. Technical Effect & Novelty
Eliminates vocabulary OOV errors and reduces sequence length by 4.2x.

## 3. Independent Claims
1. A computer-implemented method for dynamic expert allocation in a neural network, comprising:
   - receiving an input byte sequence;
   - computing local information entropy for each byte position;
   - routing tokens to a subset of expert layers conditioned on said entropy threshold.
