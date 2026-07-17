---
dataset_name: dclm_certified_v1.0
license: cc-by-4.0
language: en
task_categories: text-generation
tags:
  - iveri-certified
  - phase-8
---

# IVERI Dataset Card — dclm

## Overview

- **Dataset Name**: `dclm`
- **Version**: `1.0`
- **License**: [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Created**: 2026-07-12T12:19:04.430024
- **Source URL**: —
- **Curriculum Phases**: A

## Dataset Details

| Metric | Value |
|---|---|
| Total Documents | 60 |
| Total Size | 24,830 bytes (0.0000 GB) |
| Average Document Length | 413.8 bytes |
| Median Document Length | 413.8 bytes |
| Byte Entropy (mean) | 0.0000 bits/byte |

## Certification Pipeline

This dataset passed the **IVERI 24-stage deterministic certification pipeline** (21/24 stages).

### Removal Statistics

| Stage | Removal Rate |
|---|---|
| Deduplication | 0.00% |
| PII Detection | 0.00% |
| Safety Filtering | 0.00% |
| Quality Filtering | 0.00% |
| **Overall Retention** | **100.00%** |

### Integrity Hashes

- **SHA-256 Manifest Hash**: `N/A`
- **Freeze Certificate Hash**: `N/A`

## Statistics

Dataset statistics are stored alongside the certified data in `certified_statistics.json` and `certified_statistics.md`.

## License

This dataset is released under the [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

```bibtex
@misc{iveri_core_2026,
  title   = {IVERI CORE Dataset — dclm},
  author  = {IVERI Project},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/iveri-project/iveri-core},
}
```