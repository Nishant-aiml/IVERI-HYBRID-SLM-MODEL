# Quality Filtering & Cleaning Report

This report documents the filtering and redaction mechanisms implemented in Phase 3.0.

## 1. Quality Filters Mapped

The `QualityFilter` checks documents on several dimensions:

- **Unicode NFKC Normalization**: Converts ligatures (e.g. "ﬁle" to "file") and ensures standard character mappings.
- **Control Character Scrubbing**: Removes unicode control characters (e.g. `\x00-\x08`, `\x0b`, `\x0c`) that pollute byte sequences.
- **Broken UTF-8 Repair**: Automatically repairs invalid byte sequences using errors='replace' to prevent loading crashes.
- **Length Filtering**: Enforces min and max characters (default 100 to 100,000) to strip out empty files or giant payloads.
- **Alpha Character Ratio**: Enforces that at least 50% of the document characters belong to alphabets, filtering out raw code data blocks (when not in coding stage) or binary logs.
- **Repetition Ratio**: Rejects repetitive lines or boilerplates (repetition threshold 20%).
- **HTML Garbage Tag Density**: Rejects web pages where HTML tags exceed 20% of the document character count.
- **Emoji and Punctuation Ratios**: Filters out spammy documents with excessive symbols.

## 2. PII & Credentials Scrubbing

The `PIIRemover` redacts personal details and leaked keys using compiled regexes:

- **emails**: standard RFC 5322 regex.
- **phones**: Indian (+91/local 10 digit) and US phone formats.
- **cards**: Credit card structures (16-digit patterns).
- **IDs**: Indian Aadhaar cards (12 digits) and PAN cards (10 characters).
- **Credentials**: Leaked GitHub tokens (`ghp_`), AWS access keys (`AKIA`), AWS secret keys, OpenAI API keys (`sk-`), JWT tokens, RSA private keys, and Bearer tokens.
- **redaction**: Replaced with `[REDACTED]` placeholder.
