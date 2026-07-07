# IVERI CORE — Experiment Tracking

> **Note (Phase 3.6+):** Primary experiment provenance is stored in SQLite via `research/experiment_registry.py` (`experiments.db`). The directory layout below remains the convention for early-phase JSON-tracked runs.

## Convention

Every experiment is tracked with structured metadata inside its phase directory.

### Directory Structure

```
experiments/
├── phase_0/          Phase 0 experiments
│   └── experiment_001_infrastructure_validation/
│       ├── config.json      Frozen config used for this run
│       ├── results.json     Quantitative results
│       ├── metrics.csv      Time-series metrics (loss, lr, etc.)
│       ├── notes.md         Observations, hypotheses, conclusions
│       ├── metadata.json    Experiment metadata (date, author, git hash)
│       ├── phase.json       Phase-specific context
│       └── environment.json Hardware/software environment snapshot
├── phase_1/          Created when Phase 1 begins
├── phase_2/          Created when Phase 2 begins
└── ...
```

### Naming Convention

```
experiment_NNN_short_description/
```

- `NNN`: Zero-padded sequential number (001, 002, ...)
- `short_description`: 2-4 word snake_case description

### Required Files

#### metadata.json
```json
{
    "experiment_id": "001",
    "name": "infrastructure_validation",
    "date": "2026-06-29",
    "phase": 0,
    "step": "0.0",
    "author": "iveri-team",
    "git_hash": "abc123",
    "description": "Verify Phase 0 infrastructure is functional."
}
```

#### phase.json
```json
{
    "phase": 0,
    "task": "Project Foundation & Infrastructure",
    "research_question": "Can the project infrastructure be established in a scalable manner?",
    "dependencies": [],
    "exit_criteria": ["all tests pass", "package installable"]
}
```

#### environment.json
```json
{
    "os": "Windows 11",
    "python_version": "3.10.x",
    "torch_version": "2.3.x",
    "cuda_version": "12.1",
    "gpu": "NVIDIA RTX 3050 Laptop GPU",
    "gpu_memory_gb": 4,
    "ram_gb": 16,
    "storage": "512GB SSD"
}
```

### Guidelines

1. **Never modify** a completed experiment's files. Create a new experiment instead.
2. **Always freeze** the config used — never reference a mutable config file.
3. **Record failures** — failed experiments are as valuable as successful ones.
4. **Link experiments** — reference the parent experiment in notes.md if iterating.
5. **Be concise** — notes.md should be actionable, not verbose.
