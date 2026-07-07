# License Compliance & Compatibility Report

This report summarizes license permissions for the datasets in the registry.

## 1. Compatibility Registry

The `LicenseChecker` checks licenses against defined rules for research vs commercial compatibility:

| License | Research OK | Commercial OK | Attribution Required |
|---|---|---|---|
| MIT | Yes | Yes | Yes |
| Apache-2.0 | Yes | Yes | Yes |
| BSD-2-Clause | Yes | Yes | Yes |
| BSD-3-Clause | Yes | Yes | Yes |
| ODC-By | Yes | Yes | Yes |
| CC-BY-4.0 | Yes | Yes | Yes |
| CC-BY-SA-3.0 | Yes | Yes | Yes |
| CC-BY-NC-4.0 | Yes | No | Yes |
| NVIDIA-Open-Model | Yes | No | Yes |
| PUBLIC | Yes | Yes | No |
| PROPRIETARY | Yes | Yes | No |
| various-permissive | Yes | Yes | Yes |

## 2. Compatibility Analysis

All datasets currently in the registry are **100% compatible with research usage**.

The following datasets are restricted to **non-commercial research only**:
- `nemotron_competitive` (NVIDIA-Open-Model license)
- `opencode_instruct` (NVIDIA-Open-Model license)

All other datasets (FineWeb-Edu, DCLM, TinyStories, etc.) carry permissive licenses (MIT, Apache-2.0, ODC-By, CC-BY) allowing commercial applications.
Our proprietary Stage 3B data (`placement_qa`, `subject_explanations`) is marked as `PROPRIETARY` and restricted to authorized internal usage.
