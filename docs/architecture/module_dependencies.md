# Module Dependency Specifications

This document defines internal import constraints and class inheritance relationships for IVERI CORE.

---

## 1. Inheritance Hierarchy

All custom layers and processors must inherit from the abstract base classes in `core.interfaces`:

```
core.interfaces.BaseModule (nn.Module)
  ├── model.norms.RMSNorm
  ├── model.attention.FlashAttentionWrapper
  ├── model.mor.RecursionEngine
  │
  ├── core.interfaces.BaseRouter (BaseModule)
  │     ├── model.moe.SparseMoERouter
  │     └── model.mor.RecursionDepthRouter
  │
  ├── core.interfaces.BaseMemory (BaseModule)
  │     └── model.titans.TitansMemory
  │
  ├── core.interfaces.BaseEncoder (BaseModule)
  │     └── model.blt.BLTByteEncoder
  │
  └── core.interfaces.BaseDecoder (BaseModule)
        └── model.blt.BLTByteDecoder
```

---

## 2. Import Boundary Rules

To prevent circular dependency graphs and maintain code clarity, the following strict package-level isolation rules are enforced:

*   **`core/`** has **zero** dependencies on `configs/`, `model/`, `utils/`, or `training/`. It must remain pure, importing only standard library modules and PyTorch.
*   **`configs/`** imports from `core.exceptions` only. It has no dependencies on `model/` or `utils/`.
*   **`utils/`** imports from `core/` and `configs/`. It does not import from `model/` or any trainers.
*   **`model/`** imports from `core/` and `configs/`. Submodules within `model/` (e.g. `model/blt/`) must not cross-import each other's private components directly; communications flow only via clean, shared interfaces.
*   **`tests/`** can import from any package.
