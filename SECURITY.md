# Security Policy

This document outlines the security procedures, dependency tracking policy, and secrets management guidelines for the **IVERI CORE** research project.

## Dependency Policy

1.  **Pinning:** All dependencies in `requirements.txt` and `requirements-dev.txt` are strictly pinned or bounded using version operators to prevent breaking changes and supply chain vulnerabilities.
2.  **Audit Cadence:** Run automated dependency audits using `pip-audit` or similar tools before major phase merges.
3.  **Third-Party Code:** Avoid using unverified custom repositories or compiled binaries without visual review.

## Secrets and API Keys

1.  **Zero-Secrets Policy:** Absolutely no API keys, credentials, private paths, or Weights & Biases login tokens may be hardcoded or checked into Git.
2.  **Environment Variables:** Configuration overrides and tracking API keys (e.g. `WANDB_API_KEY`) must be loaded from local environment variables or standard `.env` files (which are ignored in `.gitignore`).
3.  **Local Logs Security:** All generated log files, profiling traces, and checkpoint weights are stored in the local directories (`logs/`, `checkpoints/`) which are blacklisted in `.gitignore`.

## Reporting a Vulnerability

If you discover a security vulnerability or package leak in this repository, please file an issue or contact the research architects directly. Do not publish vulnerabilities publicly before coordinated remediation.
