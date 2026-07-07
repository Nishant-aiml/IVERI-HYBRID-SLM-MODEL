# Stage 3B Proprietary JSON Format

Place one or more records per file under the appropriate subdirectory:

| Subdirectory | Content |
|--------------|---------|
| `university_papers/` | University exam questions and model answers |
| `gate_questions/` | GATE CS questions with step-by-step solutions |
| `placement_qa/` | Interview Q&A (DSA, systems, DBMS, etc.) |
| `subject_explanations/` | Syllabus concept explanations |

## Required fields

Every record must include:

- `id` — unique string identifier
- `license` — must be `"PROPRIETARY"`
- `language` — ISO-style code (`en`, `hi`, …)

## Text fields (one of)

**Document mode:**

```json
{
  "id": "au-cs8491-2024-q3",
  "license": "PROPRIETARY",
  "language": "en",
  "content": "Full question paper or explanation text (minimum 20 characters).",
  "metadata": {"university": "Anna University", "year": 2024}
}
```

**Q&A mode:**

```json
{
  "id": "placement-ds-001",
  "license": "PROPRIETARY",
  "language": "en",
  "question": "Explain how a hash map resolves collisions.",
  "answer": "Chaining or open addressing; chaining uses linked lists per bucket.",
  "metadata": {"topic": "data_structures"}
}
```

## Ingestion

```bash
# Validate counts only
python scripts/ingest_stage3b.py --validate-only

# Process to data/processed/stage3b/
python scripts/ingest_stage3b.py
```

Output: `stage3b_train.json`, `stage3b_val.json`, `stage3b_test.json`, and `manifest.json` (90/5/5 split, PII cleaned).

**Do not commit real proprietary content to public repositories.** Keep raw JSON gitignored or in a private store; only manifests and byte counts may appear in reports.
