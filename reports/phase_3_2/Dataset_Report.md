# Dataset Report — SFT Ingestion and In-Memory Statistics

This report documents the dataset specifications and ingestion checks.

---

## 1. Stage 2 Dataset Registry (`instruction.yaml`)

We verified the registry definitions under `data/dataset_specs/instruction.yaml`:
- **Magpie-Pro-1M**: Priority S, Apache-2.0, multi-turn messages (Weight: 30%)
- **Tulu 3 SFT Mix**: Priority S, Apache-2.0, multi-turn messages (Weight: 25%)
- **OpenHermes 2.5**: Priority A, Apache-2.0, conversations (Weight: 20%)
- **WildChat**: Priority A, Apache-2.0, conversation (Weight: 10%)
- **Code-Feedback**: Priority A, Apache-2.0, messages (Weight: 10%)
- **NuminaMath-CoT**: Priority A, Apache-2.0, alpaca (Weight: 5%)

---

## 2. Ingestion Verification Chain

The validation pipeline enforces:
1. **License Safety**: Rejects datasets lacking research-friendly licenses.
2. **Provenance**: Ensures `VERSION.json` has `stage: 2`.
3. **Integrity**: Verifies file contents against `manifest.json` SHA-256 signatures.
4. **Formatting**: Evaluates formatting using `SFTValidator` to strip empty or placeholder texts and filter records with excessive size (>50KB).
