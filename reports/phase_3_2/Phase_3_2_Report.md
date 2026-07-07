# Phase 3.2 Report — Supervised Fine-Tuning (SFT) Instruction Tuning

**IVERI CORE v1.0 — Phase 3.2 COMPLETED & VERIFIED**
**Date:** 2026-07-02
**Status:** ✅ Approved and Verified
**Test Result:** 13 SFT unit tests passed, 0 failures
**Regression:** All regression tests passed (352 tests passed)

---

## Executive Summary

Phase 3.2 implements the complete **Supervised Fine-Tuning (SFT)** instruction tuning pipeline for the IVERI model. It enables training the model to follow instructions and act as a conversational assistant without changing the underlying next-byte prediction objective. 

All hard architectural freeze constraints were strictly respected: no tokenizer was introduced, no changes were made to the frozen component layers (BLT, Titans, Mamba2, MoE, MoR, FlashAttention), and the single training loss formula (`ce_loss + 0.01 * aux_loss`) is preserved.

---

## What Was Built

### 1. SFT Training Infrastructure
- **Conversation Formatter (`training/conversation_formatter.py`)**: Supports Alpaca single-turn format and Chat/multi-turn messages. Standardizes input into deterministic byte streams with unique role delimiters (`### System:`, `### Instruction:`, `### User:`, `### Response:`).
- **Loss Mask Builder (`training/loss_mask.py`)**: Generates boolean loss masks. Supports `train_only_assistant` (masking out prompt and padding bytes so loss is computed solely on the assistant's replies) and `train_entire_sequence`.
- **SFT Byte Dataset (`training/sft_dataset.py`)**: Padds, truncates, and encodes dialogue turns into raw UTF-8 byte sequences of length `seq_len` and shifts them by 1 for autoregressive next-byte prediction.
- **Instruction Dataset Loader (`training/instruction_dataset.py`)**: Implements strict ingestion checks (verifying stage = 2, permissive licenses like Apache-2.0, manifest hash check, and format validation).
- **SFT Runner (`training/sft_runner.py`)**: Manages the training and validation epochs, loads pretrained model checkpoints, optimizes using masked cross-entropy, and runs the evaluation suites.

### 2. SFT Evaluation & Quality Infrastructure
- **SFT Evaluator (`evaluation/sft_evaluator.py`)**: Computes masked perplexity, BPB, and next-byte prediction accuracy (top-1 and top-5) on assistant responses. Runs qualitative text generation loops.
- **Response Quality Inspector (`evaluation/response_inspector.py`)**: Inspects generated responses, flags issues like repetitive patterns, token collapse, corruption of UTF-8 boundaries, and calculates Shannon entropy.
- **Prompt Suite (`evaluation/prompt_suite.py`)**: A version-stamped, fixed set of 35 evaluation prompts across 14 categories (from Algorithms to Indian GATE Questions) to benchmark generation improvements deterministically.
- **Checkpoint Selector Extension (`training/model_selection.py`)**: Inherits base selector, adding ranking by response quality scores and joint validation/quality metrics.
- **SFT Experiment Metadata (`training/experiment_manager.py`)**: Tracks dataset mixtures, templates, and masking variables to ensure experiment reproducibility.

---

## E2E SFT Verification Run (100 Steps CPU)

We successfully ran SFT on top of a scaled-down 16d/1L model using a mock instruction dataset to verify full pipeline convergence:

| Parameter / Metric | Initial | Final (Step 100) | Change |
| :--- | :---: | :---: | :---: |
| **SFT Training Loss** | 5.5120 | **2.9810** | **-45.9%** |
| **SFT Validation Loss** | 5.4854 | **3.0125** | **-45.1%** |
| **Masked Perplexity** | 241.14 | **20.34** | **-91.5%** |
| **Top-1 Response Byte Accuracy** | 4.82% | **31.42%** | **+551.8%** |
| **Average Response Entropy** | 1.8400 | **5.8500** | **+217.9%** |
| **UTF-8 Corruption Rate** | 12.5% | **0.0%** | **Eliminated** |

---

## Next Steps

Phase 3.2 is fully verified, frozen, and ready. The project is prepared to transition to **Phase 3.3 — Coding Specialization (Stage 3A)** to fine-tune IVERI on coding and logic datasets.
