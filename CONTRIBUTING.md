# Contributing to IVERI CORE

Thank you for contributing to the IVERI CORE research project!

Even if you are currently the sole developer, these guidelines ensure reproducibility, clean versioning, and rigorous quality assurance as the project scales.

## Development Workflow

1.  **Branching Policy:**
    *   `main` is the frozen, deployable release branch.
    *   Implement new phases or components in dedicated feature branches (e.g. `feat/phase-1-norms` or `refactor/validation`).
2.  **Pull Requests & Reviews:**
    *   Write a clear description of the modifications, references to target issues, and include validation results (e.g., test logs or training losses).
    *   No code should be merged into `main` without passing the complete QA routine.

## Quality Standards

Before submitting any code, run the local quality assurance suite:

```bash
python quality/run_all.py --report
```

Ensure all checks pass cleanly:
*   **Linting (Ruff):** Zero warnings or syntax errors.
*   **Formatting (Black):** Automated code styling conforming to standard configurations.
*   **Imports Sorting (isort):** Kept clean and sorted automatically.
*   **Static Type Checking (Mypy):** Verified strict types for core dataclasses and module APIs.
*   **Tests (Pytest):** 100% of unit and environment tests must succeed.

## Code Design Principles

*   **Registry Decoration:** Always register new components (e.g., norms, experts, routers) using `@register("name")`.
*   **Typing & Dataclasses:** Use strongly-typed dataclasses for hyper-parameters and explicitly annotate tensor signatures.
*   **Interface Contracts:** Custom modules must inherit from the abstract interfaces defined in `core/interfaces.py`.
*   **CPU Graceful Fallback:** Ensure all mathematical and validation operations can degrade gracefully to CPU in local testing.
