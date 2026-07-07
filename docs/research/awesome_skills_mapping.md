# Awesome Skills Mapping for IVERI CORE

This document outlines how specialized "agentic skills" from the open-source **[antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills.git)** catalog can be mapped and applied to the development, training, and deployment of **IVERI CORE**.

---

## 1. Relevant Skills & Bundles for IVERI CORE

The catalog contains over 1,600+ playbooks. Below are the specific skills mapped to IVERI's phases and architectural needs:

### A. AI/ML & PyTorch Engineering
*   **`machine-learning-ops-ml-pipeline`**
    *   *Usage in IVERI:* Guides the end-to-end MLOps pipeline for training the 10M, 50M, and 123M models.
    *   *PyTorch DDP:* Provides best practices for PyTorch Distributed Data Parallel (DDP) multi-GPU training. Mapped to **Phase 4 (Scale Incrementally)** when transitioning training to Kaggle Multi-T4 GPUs or Colab Pro.
    *   *Experiment Tracking:* Details standardized setup for Weights & Biases (`wandb`) metrics tracking and model checkpoints.

### B. Python Patterns & Quality Assurance
*   **`python-patterns` & `async-python-patterns`**
    *   *Usage in IVERI:* Establishes coding standards for the `core/` package and custom models.
    *   *PEP 8 & Formatting:* Standardizes formatting using `black` and imports layout using `isort` as verified in our local QA checks.

### C. Dependency Management
*   **`faf-wizard`**
    *   *Usage in IVERI:* Scans dependency manifest files (`pyproject.toml`, `requirements.txt`) to ensure zero-dependency leakage between modular sub-packages (e.g. keeping `model/` clean from trainer/data utility dependencies).

---

## 2. Recommended Phase Integration Checklist

| Phase | Mapped Skill / Playbook | Key Value |
|---|---|---|
| **Phase 0** | Python Formatting & Linting Playbook | Enforces Ruff rules, Black coding style, and Pytest coverage. |
| **Phase 1** | PyTorch Custom Module Patterns | Abstract base class interface compliance and registry validation patterns. |
| **Phase 2** | Dataloader Optimization Rules | High-throughput byte processing without dataloader bottlenecks. |
| **Phase 4** | PyTorch Distributed Training (DDP) | Scalable training on multiple GPUs, gradient accumulation, and checkpoint saving. |
| **Phase 5** | LLM Chat & Prompt Engineering Templates | Conversation history structures, Alpaca formatting patterns, and system instructions. |
