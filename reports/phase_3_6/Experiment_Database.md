# IVERI CORE — Experiment Database Schema

This document details the SQLite database layout used to persist metrics.

### Tables
1. **experiments:** Stores configurations, commit SHAs, purposes, and user tags.
2. **metrics:** Stores loss, perplexity, and accuracy logs per training step.
3. **hardware:** Logs peak RAM/VRAM usage, CPU load, and estimated hardware cloud costs.
4. **checkpoints:** Links parameter check files and labels the golden models.
5. **failures:** Serializes exceptions, call tracebacks, and RNG values for replaying.
6. **paper_assets:** Tracks LaTeX file outputs and Matplotlib figure formats.
