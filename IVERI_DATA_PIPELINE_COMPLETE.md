# IVERI — Complete Data Pipeline & Training Strategy
## Everything from Stage 0 to Stage 6 | June 2026 | Final Version

---

## CRITICAL: READ THIS FIRST

This document covers everything related to data for IVERI.
It supersedes all previous data discussions.

Key changes from earlier plans:
- Stage 0 (Data Engineering) added before any training
- OpenWebText replaced by FineWeb-Edu and DCLM (better quality)
- Stanford Alpaca replaced by Magpie-Pro-1M (objectively better)
- Byte-level preprocessing added (IVERI reads bytes, not tokens)
- Q+A format requirement from supervisor explicitly documented
- Three stages strictly separated (mixing them hurts the model)
- Indian engineering dataset format requirements fully specified
- All datasets verified as of June 2026

---

## WHY THREE STAGES MUST BE SEPARATE

This is what your supervisor means. Do not mix them.

```
STAGE 1 — FOUNDATION PRETRAINING
  What the model sees: raw text only
  What the model learns: next byte prediction
  Example input: "The cat sat on the mat..."
  What it learns: language, grammar, reasoning, coding basics
  Format: NO Q+A structure

STAGE 2 — INSTRUCTION TUNING (SFT)
  What the model sees: question + answer pairs
  What the model learns: how to answer questions
  Example: {"instruction": "What is TCP?", "output": "TCP is..."}
  What it learns: conversation, instruction following
  Format: Q+A format ONLY

STAGE 3 — DOMAIN SPECIALISATION
  What the model sees: domain-specific Q+A
  What the model learns: expert knowledge in your domain
  Example: GATE questions + detailed answers
  What it learns: GATE, placements, university syllabus
  Format: Q+A format ONLY

WHY SEPARATION MATTERS:
  If you mix Stage 1 raw text with Stage 2 Q+A:
    Model gets confused between prediction and answering
    Training becomes unstable
    Final quality is lower than either alone

  Correct order = foundation first, then instruct, then specialise
  Like school: learn to read first, then learn to answer questions,
  then specialise in your subject
```

---

## WHAT YOUR SUPERVISOR SPECIFICALLY SAID

Two key requirements:

### Requirement 1: Q+A Format for Fine-tuning

Every piece of fine-tuning data must be in this exact format:

```python
# FORMAT 1 — Alpaca style
{
  "instruction": "Explain the difference between TCP and UDP",
  "input": "",           # optional additional context
  "output": "TCP (Transmission Control Protocol) is..."
}

# FORMAT 2 — Conversation style (preferred for complex tasks)
{
  "messages": [
    {"role": "user",      "content": "Explain TCP vs UDP"},
    {"role": "assistant", "content": "TCP is connection-oriented..."}
  ]
}

# FORMAT 3 — Multi-turn conversation
{
  "messages": [
    {"role": "user",      "content": "What is normalisation?"},
    {"role": "assistant", "content": "Normalisation is..."},
    {"role": "user",      "content": "Can you give an example?"},
    {"role": "assistant", "content": "Sure. Consider a student table..."}
  ]
}
```

What is NOT acceptable for fine-tuning:

```
# WRONG — raw notes, not Q+A
"Normalisation is the process of organising data in a database.
 1NF: No repeating groups. 2NF: No partial dependency..."

# WRONG — textbook paragraph
"TCP was developed in the 1970s. It provides reliable, ordered,
 and error-checked delivery of a stream of octets..."

# WRONG — bullet points without question
"Difference between TCP and UDP:
 - TCP is connection-oriented
 - UDP is connectionless
 ..."
```

### Requirement 2: Domain-specific Training

General training makes a general model. You need a specialised model.
Target domain: Computer Science, Software Engineering, AI/ML,
Programming, and Indian Engineering Education (CSE/IT/CT/AI/DS).

---

## STAGE 0 — DATA ENGINEERING PIPELINE

Build this BEFORE any training. Build it ONCE. Use it for everything.

### Why Stage 0 Exists

Without a proper data pipeline:
- Training runs are not reproducible
- You cannot verify what data the model saw
- Bugs in data cause mysterious training failures
- Cannot compare experiments fairly
- Legal issues from unlicensed data

### Complete Stage 0 Checklist

```
□ 0.1  Dataset Downloader
□ 0.2  License Verification
□ 0.3  UTF-8 Validation + Byte Encoding
□ 0.4  Deduplication
□ 0.5  Language Detection
□ 0.6  Quality Filtering
□ 0.7  PII Removal
□ 0.8  Toxicity Filtering
□ 0.9  Train/Validation/Test Splitting
□ 0.10 Dataset Versioning
□ 0.11 Metadata Generation
□ 0.12 Token/Byte Counting
□ 0.13 Mixing Weights Configuration
□ 0.14 SFT Format Validator
□ 0.15 Conversation Format Validator
□ 0.16 Dataset Statistics Report
```

---

### 0.1 — Dataset Downloader

What it does: Downloads datasets from HuggingFace and other sources.
Saves locally with consistent naming.

```python
# data/pipeline/downloader.py

from datasets import load_dataset
import os
import json
from datetime import datetime

DATASETS = {
    # Stage 1 — Foundation
    "tinystories": {
        "hf_name": "roneneldan/TinyStories",
        "split": "train",
        "priority": "S",
        "stage": 1,
        "license": "MIT",
        "format": "pretrain"
    },
    "fineweb_edu": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",  # start with 10B token sample
        "priority": "S",
        "stage": 1,
        "license": "ODC-By",
        "format": "pretrain"
    },
    "dclm_baseline": {
        "hf_name": "mlfoundations/dclm-baseline-1.0",
        "priority": "S",
        "stage": 1,
        "license": "ODC-By",
        "format": "pretrain"
    },
    "wikipedia": {
        "hf_name": "wikimedia/wikipedia",
        "config": "20220301.en",
        "priority": "A",
        "stage": 1,
        "license": "CC-BY-SA-3.0",
        "format": "pretrain"
    },
    "finemath": {
        "hf_name": "HuggingFaceFW/finemath",
        "priority": "A",
        "stage": 1,
        "license": "ODC-By",
        "format": "pretrain"
    },
    # Stage 2 — Instruction Tuning
    "magpie_pro": {
        "hf_name": "magpie-align/Magpie-Pro-1M-v0.1",
        "priority": "S",
        "stage": 2,
        "license": "Apache-2.0",
        "format": "sft"
    },
    "tulu3_sft": {
        "hf_name": "allenai/tulu-3-sft-mixture",
        "priority": "S",
        "stage": 2,
        "license": "Apache-2.0",
        "format": "sft"
    },
    "openhermes": {
        "hf_name": "teknium/OpenHermes-2.5",
        "priority": "A",
        "stage": 2,
        "license": "Apache-2.0",
        "format": "sft"
    },
    # Stage 3A — Coding
    "the_stack_v2_python": {
        "hf_name": "bigcode/the-stack-v2",
        "config": "Python",
        "priority": "S",
        "stage": "3A",
        "license": "various-permissive",
        "format": "pretrain"
    },
    "opencode_instruct": {
        "hf_name": "nvidia/OpenCodeInstruct",
        "priority": "S",
        "stage": "3A",
        "license": "NVIDIA-Open-Model",
        "format": "sft"
    },
    "leetcode_dataset": {
        "hf_name": "yunhui/LeetCodeDataset",
        "priority": "S",
        "stage": "3A",
        "license": "CC-BY-4.0",
        "format": "sft"
    },
}

def download_dataset(name, save_dir="data/raw"):
    info = DATASETS[name]
    print(f"Downloading {name}...")
    
    kwargs = {"path": info["hf_name"]}
    if "config" in info:
        kwargs["name"] = info["config"]
    if "split" in info:
        kwargs["split"] = info["split"]
    
    ds = load_dataset(**kwargs)
    
    save_path = os.path.join(save_dir, name)
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)
    
    # save metadata
    metadata = {
        "name": name,
        "downloaded_at": datetime.now().isoformat(),
        "info": info,
        "num_rows": len(ds) if hasattr(ds, '__len__') else "unknown"
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {name} to {save_path}")
    return ds
```

---

### 0.2 — License Verification

What it does: Ensures every dataset you use is legally permitted.
Critical before any commercial or public use.

```python
# data/pipeline/license_checker.py

LICENSE_COMPATIBILITY = {
    # Format: license -> [research_ok, commercial_ok, attribution_required]
    "MIT":           [True,  True,  False],
    "Apache-2.0":    [True,  True,  False],
    "CC-BY-4.0":     [True,  True,  True],
    "CC-BY-SA-3.0":  [True,  True,  True],  # Wikipedia
    "ODC-By":        [True,  True,  True],   # FineWeb, DCLM
    "CC-BY-NC-4.0":  [True,  False, True],  # research only
    "research-only": [True,  False, False],
    "custom":        [True,  False, False],  # check individually
}

LICENSE_REGISTRY = {
    "TinyStories":        "MIT",
    "FineWeb-Edu":        "ODC-By",
    "DCLM-Baseline":      "ODC-By",
    "Wikipedia":          "CC-BY-SA-3.0",
    "FineMath":           "ODC-By",
    "SlimPajama":         "Apache-2.0",
    "Magpie-Pro-1M":      "Apache-2.0",
    "Tulu-3-SFT":         "Apache-2.0",
    "OpenHermes-2.5":     "Apache-2.0",
    "WildChat":           "Apache-2.0",
    "UltraFeedback":      "MIT",
    "The Stack v2":       "various",   # filter by license column
    "OpenCodeInstruct":   "NVIDIA-Open-Model",  # check terms
    "LeetCodeDataset":    "CC-BY-4.0",
    "Code-Feedback":      "Apache-2.0",
    "Codeforces+R1":      "CC-BY-4.0",
    "NuminaMath":         "Apache-2.0",
    "GATE Papers":        "PUBLIC",    # government documents
    "University Papers":  "PUBLIC",    # educational institutions
}

def verify_license(dataset_name, use_case="research"):
    license = LICENSE_REGISTRY.get(dataset_name, "unknown")
    if license == "unknown":
        print(f"WARNING: Unknown license for {dataset_name}")
        return False
    if license == "various":
        print(f"NOTE: {dataset_name} has mixed licenses. Filter by license column.")
        return True
    if license in ["PUBLIC", "government"]:
        return True
    compat = LICENSE_COMPATIBILITY.get(license, [False, False, False])
    if use_case == "research":
        return compat[0]
    if use_case == "commercial":
        return compat[1]
    return False
```

---

### 0.3 — UTF-8 Validation + Byte Encoding

This is unique to IVERI. Standard LLM pipelines use tokenizers.
IVERI reads raw bytes — so byte encoding is critical.

```python
# data/pipeline/byte_encoder.py

import torch
import numpy as np
from typing import List

def validate_utf8(text: str) -> bool:
    """Check if text can be encoded as valid UTF-8."""
    try:
        text.encode('utf-8').decode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False

def text_to_bytes(text: str) -> List[int]:
    """Convert text to list of byte values (0-255)."""
    return list(text.encode('utf-8'))

def bytes_to_tensor(byte_list: List[int], 
                     seq_len: int = 256,
                     pad_value: int = 0) -> torch.Tensor:
    """Convert byte list to fixed-length tensor."""
    if len(byte_list) > seq_len:
        byte_list = byte_list[:seq_len]
    else:
        byte_list = byte_list + [pad_value] * (seq_len - len(byte_list))
    return torch.tensor(byte_list, dtype=torch.long)

def validate_byte_range(byte_list: List[int]) -> bool:
    """Ensure all bytes are in valid range 0-255."""
    return all(0 <= b <= 255 for b in byte_list)

# IMPORTANT: Byte length vs token length
# English: ~1 byte per character, ~4 chars per word
#   seq_len=256 bytes ≈ 50-64 English words
# Chinese: ~3 bytes per character
#   seq_len=256 bytes ≈ 85 Chinese characters
# Code: variable, depends on syntax
#   seq_len=256 bytes ≈ 3-5 lines of Python

# For IVERI training, adjust seq_len accordingly:
# Early training (sanity):  seq_len = 256
# Normal training:           seq_len = 512
# Long context testing:      seq_len = 2048+
```

---

### 0.4 — Deduplication

What it does: Removes near-duplicate documents.
Why: duplicates cause models to memorise specific text rather than learn patterns.

```python
# data/pipeline/deduplication.py
# Uses MinHash for approximate near-duplicate detection

from datasketch import MinHash, MinHashLSH
import hashlib
from typing import List, Set

def exact_deduplicate(texts: List[str]) -> List[str]:
    """Remove exact duplicates using MD5 hash."""
    seen = set()
    unique = []
    for text in texts:
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(text)
    return unique

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash signature for approximate matching."""
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode('utf-8'))
    return m

def near_deduplicate(texts: List[str], 
                      threshold: float = 0.8,
                      num_perm: int = 128) -> List[str]:
    """Remove near-duplicates using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_texts = []
    
    for i, text in enumerate(texts):
        m = get_minhash(text, num_perm)
        key = f"doc_{i}"
        
        try:
            result = lsh.query(m)
            if not result:  # no similar document found
                lsh.insert(key, m)
                unique_texts.append(text)
        except Exception:
            unique_texts.append(text)
    
    return unique_texts

# Install: pip install datasketch
```

---

### 0.5 — Language Detection

What it does: Identifies the language of each document.
Why: IVERI's target audience is English (with some Hindi later).
Remove non-target languages from Stage 1-3.

```python
# data/pipeline/language_detector.py

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42  # reproducible results

ALLOWED_LANGUAGES_STAGE1 = {"en"}           # English only for start
ALLOWED_LANGUAGES_STAGE3B = {"en", "hi"}    # English + Hindi for Indian dataset

def detect_language(text: str) -> str:
    """Detect language of text. Returns ISO 639-1 code."""
    try:
        if len(text.strip()) < 20:
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"

def filter_by_language(texts: List[str], 
                        allowed: set = ALLOWED_LANGUAGES_STAGE1) -> List[str]:
    """Keep only documents in allowed languages."""
    filtered = []
    for text in texts:
        lang = detect_language(text)
        if lang in allowed:
            filtered.append(text)
    return filtered

# Install: pip install langdetect
```

---

### 0.6 — Quality Filtering

What it does: Removes low-quality text (too short, gibberish, spam).

```python
# data/pipeline/quality_filter.py

import re
from typing import Callable, List

def min_length_filter(text: str, min_chars: int = 100) -> bool:
    """Remove very short documents."""
    return len(text.strip()) >= min_chars

def max_length_filter(text: str, max_chars: int = 100000) -> bool:
    """Remove extremely long documents (likely data dumps)."""
    return len(text.strip()) <= max_chars

def alpha_ratio_filter(text: str, min_ratio: float = 0.5) -> bool:
    """Ensure enough alphabetic characters (not just symbols/numbers)."""
    if not text:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count / len(text) >= min_ratio

def line_length_filter(text: str, 
                        max_avg_line_length: int = 1000) -> bool:
    """Remove documents with very long lines (likely minified code/data)."""
    lines = text.split('\n')
    if not lines:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    return avg_len <= max_avg_line_length

def repetition_filter(text: str, max_rep_ratio: float = 0.2) -> bool:
    """Remove documents with high word repetition."""
    words = text.lower().split()
    if len(words) < 10:
        return True
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio >= (1 - max_rep_ratio)

def apply_quality_filters(texts: List[str]) -> List[str]:
    """Apply all quality filters."""
    filters = [
        min_length_filter,
        max_length_filter,
        alpha_ratio_filter,
        line_length_filter,
        repetition_filter,
    ]
    
    filtered = []
    for text in texts:
        if all(f(text) for f in filters):
            filtered.append(text)
    
    removal_rate = 1 - len(filtered) / max(len(texts), 1)
    print(f"Quality filter: kept {len(filtered)}/{len(texts)} "
          f"({removal_rate:.1%} removed)")
    return filtered
```

---

### 0.7 — PII Removal

What it does: Removes Personally Identifiable Information.
Why: Legal requirement (GDPR), prevents model from memorising private data.

```python
# data/pipeline/pii_remover.py

import re

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_india": r'\b[6-9]\d{9}\b',           # Indian mobile numbers
    "phone_us": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "aadhaar": r'\b\d{4}\s?\d{4}\s?\d{4}\b',    # Indian Aadhaar
    "pan": r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',        # Indian PAN card
    "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
}

def remove_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Remove PII from text, replace with placeholder."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return text

def has_pii(text: str) -> bool:
    """Check if text contains PII."""
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True
    return False
```

---

### 0.8 — Train/Validation/Test Splitting

```python
# data/pipeline/splitter.py

import random
from typing import Tuple, List

def split_dataset(data: List, 
                   train_ratio: float = 0.98,
                   val_ratio: float = 0.01,
                   test_ratio: float = 0.01,
                   seed: int = 42) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test.
    Default: 98% train, 1% val, 1% test.
    
    For small datasets (Stage 3B proprietary):
    Use 90% / 5% / 5% instead.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    data_copy = list(data)
    random.shuffle(data_copy)
    
    n = len(data_copy)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    
    train = data_copy[:n_train]
    val   = data_copy[n_train:n_train + n_val]
    test  = data_copy[n_train + n_val:]
    
    print(f"Split: {len(train)} train | {len(val)} val | {len(test)} test")
    return train, val, test

# IMPORTANT: Create splits BEFORE any training
# Use the SAME test set for ALL experiments
# Never train on test set (contamination)
# Val set = used during training to monitor
# Test set = used ONLY for final evaluation
```

---

### 0.9 — Dataset Versioning

```python
# data/pipeline/versioning.py

import json
import hashlib
import os
from datetime import datetime

def create_dataset_version(dataset_name: str,
                             data_path: str,
                             config: dict) -> str:
    """
    Create a version stamp for a dataset.
    Returns version ID string.
    """
    # hash the config to detect changes
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    timestamp = datetime.now().strftime("%Y%m%d")
    version_id = f"{dataset_name}_v{timestamp}_{config_hash}"
    
    version_info = {
        "version_id": version_id,
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "config_hash": config_hash,
        "data_path": data_path,
    }
    
    version_file = os.path.join(data_path, "VERSION.json")
    with open(version_file, "w") as f:
        json.dump(version_info, f, indent=2)
    
    print(f"Dataset version: {version_id}")
    return version_id

# RULE: Every training run must log which dataset version it used
# This makes experiments reproducible
```

---

### 0.10 — Token/Byte Counting

Critical for IVERI since it processes bytes, not tokens.

```python
# data/pipeline/byte_counter.py

def count_bytes(text: str) -> int:
    """Count UTF-8 bytes in text."""
    return len(text.encode('utf-8'))

def count_bytes_dataset(texts: List[str]) -> dict:
    """Count total bytes in a dataset."""
    byte_counts = [count_bytes(t) for t in texts]
    return {
        "total_bytes": sum(byte_counts),
        "total_gb": sum(byte_counts) / (1024**3),
        "avg_bytes_per_doc": sum(byte_counts) / len(byte_counts),
        "min_bytes": min(byte_counts),
        "max_bytes": max(byte_counts),
        "num_documents": len(texts),
    }

# BYTE vs TOKEN comparison for IVERI:
# Standard tokenizer: 1 token ≈ 4 bytes (English)
# IVERI: works directly in bytes
#
# If a paper says "10 billion tokens":
# ≈ 40 billion bytes for English text
#
# Your training budget:
# 2 billion tokens = 8 billion bytes = ~8GB
# 10 billion tokens = 40 billion bytes = ~40GB
```

---

### 0.11 — SFT Format Validator

This is specifically what your supervisor requires.
Every fine-tuning sample MUST pass this validation.

```python
# data/pipeline/sft_validator.py

from typing import List, Dict, Any
import json

def validate_sft_sample_alpaca(sample: dict) -> tuple:
    """
    Validate Alpaca-style Q+A format.
    Returns (is_valid, error_message)
    """
    required = ["instruction", "output"]
    
    for field in required:
        if field not in sample:
            return False, f"Missing required field: {field}"
    
    if not isinstance(sample["instruction"], str):
        return False, "instruction must be a string"
    
    if not isinstance(sample["output"], str):
        return False, "output must be a string"
    
    if len(sample["instruction"].strip()) < 5:
        return False, "instruction too short (< 5 chars)"
    
    if len(sample["output"].strip()) < 10:
        return False, "output too short (< 10 chars)"
    
    # check for empty or placeholder answers
    bad_outputs = [
        "TODO", "PLACEHOLDER", "...", "N/A", 
        "TBD", "Answer here"
    ]
    for bad in bad_outputs:
        if sample["output"].strip() == bad:
            return False, f"output contains placeholder: {bad}"
    
    return True, "OK"

def validate_sft_sample_conversation(sample: dict) -> tuple:
    """
    Validate conversation/messages format.
    Returns (is_valid, error_message)
    """
    if "messages" not in sample:
        return False, "Missing 'messages' field"
    
    messages = sample["messages"]
    
    if not isinstance(messages, list):
        return False, "messages must be a list"
    
    if len(messages) < 2:
        return False, "must have at least 2 messages (user + assistant)"
    
    valid_roles = {"user", "assistant", "system"}
    for i, msg in enumerate(messages):
        if "role" not in msg:
            return False, f"message {i} missing 'role'"
        if "content" not in msg:
            return False, f"message {i} missing 'content'"
        if msg["role"] not in valid_roles:
            return False, f"message {i} invalid role: {msg['role']}"
        if len(msg["content"].strip()) < 2:
            return False, f"message {i} content too short"
    
    # must start with user (or system then user)
    first_non_system = next(
        (m for m in messages if m["role"] != "system"), None
    )
    if first_non_system and first_non_system["role"] != "user":
        return False, "first message must be from user"
    
    # must end with assistant
    if messages[-1]["role"] != "assistant":
        return False, "last message must be from assistant"
    
    return True, "OK"

def validate_sft_dataset(dataset: List[dict]) -> dict:
    """
    Validate entire SFT dataset.
    Returns statistics and list of invalid samples.
    """
    valid = 0
    invalid = 0
    errors = []
    
    for i, sample in enumerate(dataset):
        # try both formats
        if "messages" in sample:
            is_valid, error = validate_sft_sample_conversation(sample)
        else:
            is_valid, error = validate_sft_sample_alpaca(sample)
        
        if is_valid:
            valid += 1
        else:
            invalid += 1
            errors.append({"index": i, "error": error, 
                           "sample": str(sample)[:100]})
    
    return {
        "total": len(dataset),
        "valid": valid,
        "invalid": invalid,
        "invalid_rate": invalid / max(len(dataset), 1),
        "errors": errors[:20],  # show first 20 errors
        "status": "PASS" if invalid == 0 else "FAIL"
    }

# HOW TO USE:
# Before any fine-tuning run, call:
# results = validate_sft_dataset(your_dataset)
# if results["status"] == "FAIL":
#     print(results["errors"])
#     # fix the bad samples before training
```

---

### 0.12 — Dataset Statistics Report

```python
# data/pipeline/statistics.py

def generate_dataset_report(name: str, 
                              data: List[str],
                              stage: str) -> str:
    """
    Generate a statistics report for a dataset.
    Print this before every training run.
    """
    byte_stats = count_bytes_dataset(data)
    
    lengths = [len(d.split()) for d in data]
    
    report = f"""
=== DATASET REPORT: {name} ===
Stage:           {stage}
Documents:       {byte_stats['num_documents']:,}
Total GB:        {byte_stats['total_gb']:.2f} GB
Avg bytes/doc:   {byte_stats['avg_bytes_per_doc']:.0f}
Avg words/doc:   {sum(lengths)/len(lengths):.0f}
Min words:       {min(lengths)}
Max words:       {max(lengths)}
Generated:       {datetime.now().isoformat()}
================================
"""
    return report
```

---

## STAGE 1 — FOUNDATION PRETRAINING

### What This Stage Does

Teaches IVERI language, grammar, reasoning, coding fundamentals.
Model learns by predicting the next byte.
No Q+A structure. Just raw text.

### Why FineWeb-Edu Not OpenWebText

```
OpenWebText (2019):
  Filtered Reddit-shared links
  Good in 2019-2022
  Superseded by better alternatives
  No longer recommended for 2026

FineWeb-Edu (2024-2025):
  15 trillion tokens of web text
  Filtered by Llama-3-70B for educational quality
  Shows dramatically better reasoning performance
  Research confirms: educational subsets improve
  ALL capabilities, not just knowledge recall
  Use this instead of OpenWebText
```

### Dataset Order (Train in This Order)

```
WEEK 1-2: TinyStories only
  Why: simplest possible text
       confirms model trains (loss goes down)
       confirms no NaN
       confirms Titans stable after step 500
  Size: 2GB, fast to train
  If this fails: do NOT proceed to bigger data

WEEK 3-4: Add FineWeb-Edu (10B token sample)
  Why: high quality filtered web text
       educational content improves reasoning
  Size: ~40GB for 10B token sample
  Use: HuggingFaceFW/fineweb-edu (sample-10BT)

WEEK 5-6: Add DCLM-Baseline (subset)
  Why: best benchmark performance of any web corpus
       complements FineWeb-Edu
  Size: use 30-50GB subset initially
  Use: mlfoundations/dclm-baseline-1.0

WEEK 7-8: Add Wikipedia + FineMath
  Wikipedia: factual knowledge, structured
  FineMath: 54B tokens of math
            math training improves ALL reasoning
            not just math questions
  Use: wikimedia/wikipedia + HuggingFaceFW/finemath

WEEK 9+: Add The Stack v2 (code)
  Mix code into pretraining
  Start with Python only
  Add Java, C++, JavaScript later
  Why: code improves general reasoning (proven)
  Use: bigcode/the-stack-v2 (Python config)
```

### Mixing Weights

```python
# configs/data_config.py

STAGE1_MIXING_WEIGHTS = {
    # what fraction of each batch comes from each dataset
    "tinystories":    0.05,   # 5%  — keep some simple text
    "fineweb_edu":    0.35,   # 35% — primary web text
    "dclm_baseline":  0.25,   # 25% — secondary web text
    "wikipedia":      0.10,   # 10% — factual knowledge
    "finemath":       0.10,   # 10% — math reasoning
    "the_stack_v2":   0.15,   # 15% — code
}
# These sum to 1.0
# Adjust based on what your validation loss shows
```

### Hardware and Duration

```
10M model — RTX 3050 laptop
  seq_len=256, batch_size=8
  ~2 days for TinyStories pass
  ~1 week for 1B byte pass of FineWeb-Edu

35M model — RTX 3050 + Kaggle T4
  seq_len=512, batch_size=4
  ~1 week per billion tokens

70M-300M — Kaggle T4 x2 (free) or Colab Pro
  A100 on Colab Pro: ~4x faster than T4
  Cost: ~Rs 800-1500/month for Colab Pro
```

---

## STAGE 2 — SUPERVISED FINE-TUNING (SFT)

### What Your Supervisor Specifically Means

This is the stage where every sample must be in Q+A format.

Raw pretraining text teaches the model to continue text.
SFT teaches the model to ANSWER questions.

Without SFT, if you ask "What is TCP?", the model writes:
"TCP is a protocol. TCP was developed in 1974. TCP provides..."
(continues writing ABOUT TCP, doesn't answer the question)

With SFT, the model says:
"TCP stands for Transmission Control Protocol. It ensures..."
(actually answers the question like a helpful assistant)

### Why Magpie Not Stanford Alpaca

```
Stanford Alpaca (2022):
  52,000 examples
  Generated by GPT-3 (outdated)
  Simple instructions only
  Many low-quality responses
  Superseded by much better alternatives

Magpie-Pro-1M (ICLR 2025, accepted paper):
  1,000,000 examples
  Generated by Llama-3-Instruct (2024 model)
  Diverse, complex, multi-turn capable
  Research result: Magpie models outperform
  official Llama-3-8B-Instruct trained on
  10 MILLION examples
  Use Magpie. Skip Alpaca.
```

### Datasets in Priority Order

```
PRIORITY S — USE THESE (non-negotiable)

Magpie-Pro-1M
  HuggingFace: magpie-align/Magpie-Pro-1M-v0.1
  Size: 1 million conversations
  License: Apache-2.0 (commercial OK)
  Format: messages (user/assistant)
  Contains: general, coding, reasoning, creative

Tulu 3 SFT Mix (AI2, 2025)
  HuggingFace: allenai/tulu-3-sft-mixture
  Size: 940,000 examples
  License: Apache-2.0
  Format: messages
  Contains: diverse tasks, state of the art

PRIORITY A — ADD THESE

OpenHermes 2.5
  HuggingFace: teknium/OpenHermes-2.5
  Size: 1 million conversations
  License: Apache-2.0
  Contains: diverse, high quality

WildChat
  HuggingFace: allenai/WildChat
  Size: 652,000 real human-AI conversations
  License: Apache-2.0
  Contains: real user queries, diverse topics
  Why: real human queries, not synthetic

PRIORITY A (CODING SPECIFIC)

Code-Feedback
  HuggingFace: m-a-p/Code-Feedback
  Size: large
  License: Apache-2.0
  Contains: iterative debugging conversations
  Why: teaches step-by-step code fixing

OpenCodeInstruct
  HuggingFace: nvidia/OpenCodeInstruct
  Size: large
  License: NVIDIA Open Model License
  Contains: code instruction-following

PRIORITY A (MATH/REASONING)

NuminaMath-CoT
  HuggingFace: AI-MO/NuminaMath-CoT
  Size: 860,000 problems
  License: Apache-2.0
  Contains: math with chain-of-thought reasoning

LATER (Preference Alignment — Stage 4)

UltraFeedback
  HuggingFace: openbmb/UltraFeedback
  Size: 64,000 prompts, 4 responses each, GPT-4 ranked
  Use for DPO/RLHF after SFT is stable
  Do NOT use during SFT stage
```

### SFT Data Format Examples

```python
# Every sample in Stage 2 MUST look like one of these:

# FORMAT 1: Simple instruction-answer
{
  "instruction": "Explain the difference between stack and queue",
  "output": "Stack uses LIFO (Last In First Out)..."
}

# FORMAT 2: Instruction with context
{
  "instruction": "Fix the bug in this Python code",
  "input": "def add(a, b):\n    return a - b",
  "output": "The bug is on line 2. You used subtraction (-) instead..."
}

# FORMAT 3: Conversation (preferred for complex tasks)
{
  "messages": [
    {"role": "system",    "content": "You are a helpful CS tutor"},
    {"role": "user",      "content": "What is recursion?"},
    {"role": "assistant", "content": "Recursion is a function that calls itself..."},
    {"role": "user",      "content": "Can you show me an example?"},
    {"role": "assistant", "content": "Sure. Here is factorial in Python:\n```python\ndef factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)\n```"}
  ]
}
```

---

## STAGE 3A — CODING SPECIALISATION

### Datasets

```
PRIORITY S

The Stack v2
  HuggingFace: bigcode/the-stack-v2
  Size: 6TB total, use Python subset (~100GB)
  License: various — FILTER by license column
           only use "MIT", "Apache-2.0", "BSD"
  Languages to include: Python, C, C++, Java, JavaScript, Go
  Format: raw code files (pretrain style)
  Also use: bigcode/the-stack-v2-train-smol-ids for smaller start

Nemotron-SFT-Competitive-Programming-v2 (March 2026)
  HuggingFace: nvidia/Nemotron-SFT-Competitive-Programming-v2
  Released: March 2026 — most recent competitive dataset
  Contains: competitive programming with DeepSeek-R1 reasoning traces
  Why unique: shows HOW to think through hard problems step by step
  License: NVIDIA Open Model License
  Format: instruction + reasoning trace + solution

LeetCodeDataset (2025)
  HuggingFace: yunhui/LeetCodeDataset
  Contains: all LeetCode Python problems
  Key feature: temporal splits (pre/post July 2024)
              prevents contamination in evaluation
  100+ test cases per problem for verification
  License: CC-BY-4.0
  Format: Q+A with test cases

PRIORITY A

OpenCodeInstruct
  HuggingFace: nvidia/OpenCodeInstruct
  Large-scale coding instruction SFT data

Code-Feedback
  HuggingFace: m-a-p/Code-Feedback
  Iterative debugging conversations
  Teaches: debug this → still wrong → try again

CodeSearchNet
  HuggingFace: code_search_net
  Code + documentation pairs
  Teaches: what code MEANS, not just syntax

Codeforces + DeepSeek-R1 Solutions
  HuggingFace: open-r1/codeforces
  10K+ Codeforces problems + ~100K reasoning solutions
  License: CC-BY-4.0

NuminaMath-CoT
  Math with reasoning traces
  Math improves general programming problem solving
```

### Coding Data Format

```python
# For The Stack v2 (pretrain style — Stage 1 style mixed in)
{
  "content": "def binary_search(arr, target):\n    left, right = 0, len(arr)-1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
  "language": "Python",
  "license": "MIT"
}

# For LeetCode / competitive (Q+A style)
{
  "instruction": "Given an array of integers and a target, return indices of two numbers that add up to target. LeetCode #1",
  "input": "nums = [2, 7, 11, 15], target = 9",
  "output": "Use a hashmap to store complements.\n\n```python\ndef twoSum(self, nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target-n], i]\n        seen[n] = i\n```\n\nTime: O(n), Space: O(n)"
}

# For debugging (Code-Feedback style)
{
  "messages": [
    {"role": "user", "content": "Fix this code:\ndef reverse_string(s):\n    for i in range(len(s)):\n        s[i] = s[-i]\n    return s"},
    {"role": "assistant", "content": "Two issues:\n1. Strings are immutable in Python\n2. Index -0 == 0, causing incorrect reversal\n\nFix:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nOr if you need in-place: convert to list first."},
    {"role": "user", "content": "What if input is a list?"},
    {"role": "assistant", "content": "If input is a list, use two-pointer approach:\n```python\ndef reverse_list(lst):\n    left, right = 0, len(lst)-1\n    while left < right:\n        lst[left], lst[right] = lst[right], lst[left]\n        left += 1\n        right -= 1\n    return lst\n```\nThis is O(n) time, O(1) space."}
  ]
}
```

---

## STAGE 3B — IVERI PROPRIETARY DATASET

### Why This Is Your Competitive Moat

```
ChatGPT training data: general internet
LLaMA training data: general internet
IVERI Stage 3B data: YOU BUILD IT

Nobody has:
  Anna University question papers in Q+A format
  GATE CS papers with detailed explanations
  Indian placement interview patterns
  VTU/AKTU/GTU specific syllabus questions
  Viva questions for Indian practicals
  Project guides for Indian final year students

This data does not exist anywhere.
ChatGPT cannot answer "Explain normalisation
for Anna University Unit 3 format" as well as
a model trained specifically on Anna University
question papers and mark schemes.

This is your unfair advantage.
Start collecting NOW — parallel to architecture work.
```

### Complete List of What to Build

```
CATEGORY 1: University Exam Papers
  Sources: Anna University, VTU, AKTU, SRM, GTU, RTU,
           Mumbai University, Pune University
  Years: 2015-2025 (10 years of papers)
  Branches: CSE, IT, CT, AI/DS, ECE (for domain)
  Format: Q+A (see below)
  Estimated size: 10,000+ Q+A pairs

CATEGORY 2: GATE CS Previous Years
  Source: GATE official website + Made Easy solutions
  Years: 1991-2025 (all available)
  Subjects: DS, Algorithms, OS, DBMS, CN, TOC,
            Compilers, Digital Logic, Architecture,
            Programming, Aptitude
  Format: question + detailed explanation + answer
  Estimated size: 5,000+ Q+A pairs

CATEGORY 3: GATE DA (Data Analytics) — New paper
  Source: GATE 2024-2025 DA papers
  Subjects: Statistics, ML basics, Data Analysis
  Less coverage = bigger advantage for IVERI
  Estimated size: 500+ Q+A pairs

CATEGORY 4: Placement Interview Q+A
  Companies: TCS NQT, Infosys, Wipro, Cognizant,
             Capgemini, HCL, Accenture,
             Amazon, Flipkart, Swiggy, Zomato,
             Juspay, CRED (startups)
  Types: Technical round 1, Technical round 2,
         Managerial, HR
  Topics: DSA, DBMS, OS, CN, OOP, System Design,
          Problem solving, Aptitude
  Format: multi-turn conversation (realistic)
  Estimated size: 20,000+ pairs

CATEGORY 5: Subject-wise Explanations
  For every subject in CSE/IT/AI/DS curriculum:
    Data Structures
    Algorithms
    DBMS
    Operating Systems
    Computer Networks
    Software Engineering
    OOP with Java/Python
    Machine Learning
    Deep Learning
    NLP
    Computer Vision
    Cloud Computing
    Cybersecurity
    Web Development
    Mobile Development
  Format: concept question → explanation with examples
  Estimated size: 50,000+ Q+A pairs

CATEGORY 6: Viva Questions
  Source: Your own lab experience + online forums
  Subjects: all lab subjects (DS Lab, DBMS Lab,
            Networks Lab, ML Lab, etc.)
  Format: expected viva Q+A
  Why valuable: students are terrified of vivas
  Estimated size: 5,000+ pairs

CATEGORY 7: Lab Manual Procedures
  Format: "Write procedure for [experiment name]" → procedure
  Sources: your university lab manuals
  Estimated size: 1,000+ pairs

CATEGORY 8: Assignment Questions + Solutions
  Format: assignment question → solution with explanation
  Across all subjects
  Estimated size: 5,000+ pairs

CATEGORY 9: Final Year Project Guides
  Format:
    "Suggest final year projects for CSE student
     interested in ML" → list with details
    "How to implement face recognition attendance
     system" → step-by-step guide
  Topics: AI/ML, IoT, Web, Blockchain, Cloud,
           Cybersecurity, AR/VR
  Estimated size: 2,000+ guides

CATEGORY 10: Mini-Project Documentation
  Format: complete project descriptions with
          architecture, modules, tech stack
  Estimated size: 500+ complete mini-projects

CATEGORY 11: Semester Notes → Q+A
  Take existing notes, convert EVERY concept to Q+A
  Do NOT put raw notes in training data
  Every concept = "Explain X" → answer
  Estimated size: 30,000+ pairs

CATEGORY 12: Debugging Examples
  Common mistakes Indian CS students make
  Format: buggy code → explanation of bug → fixed code
  Languages: C, C++, Java, Python
  Topics: pointers, recursion, sorting, linked lists
  Estimated size: 5,000+ pairs

CATEGORY 13: Resume + Career Guidance
  Indian-specific placement advice
  Format: Q+A about resume writing, aptitude, HR
  Example: "How to write resume for TCS NQT?" → detailed answer
  Estimated size: 1,000+ pairs

CATEGORY 14: Research Paper Q+A
  For AI/DS students specifically
  arXiv papers summarised as Q+A
  "What is the key contribution of Attention Is All You Need?"
  Estimated size: 2,000+ papers

TOTAL ESTIMATED: ~130,000+ Q+A pairs
                 This is the core of IVERI's advantage
```

### Exact Format for Every Sample

Every single sample in Stage 3B must follow ONE of these formats.
No exceptions. Your supervisor is explicit about this.

```python
# EXAMPLE 1: University Exam Q+A
{
  "instruction": "Explain Second Normal Form (2NF) with an example. [Anna University, Nov 2023, 8 marks]",
  "output": "Second Normal Form (2NF):\n\nDefinition: A relation is in 2NF if it is in 1NF and every non-prime attribute is fully functionally dependent on the primary key.\n\nKey concept: No partial dependency (non-key attribute depending on part of a composite key).\n\nExample:\n\nBad table (not 2NF):\nEnrollment(StudentID, CourseID, StudentName, CourseName, Grade)\nPrimary Key: (StudentID, CourseID)\n\nProblems:\n- StudentName depends only on StudentID (partial)\n- CourseName depends only on CourseID (partial)\n- Only Grade depends on full key\n\nFixed to 2NF:\nStudent(StudentID, StudentName)\nCourse(CourseID, CourseName)\nEnrollment(StudentID, CourseID, Grade)\n\nFor 8-mark answer: Always draw before and after tables. Identify the partial dependency. Show the decomposed tables. State why each is now 2NF."
}

# EXAMPLE 2: GATE Question
{
  "instruction": "GATE CS 2023: Consider a hash table with 10 slots. Collisions are resolved using chaining. Keys 5, 28, 19, 15, 20, 33, 12, 17, 10 are inserted. The load factor is?",
  "output": "Load factor α = n/m where n = number of keys, m = number of slots.\n\nn = 9 keys (count: 5,28,19,15,20,33,12,17,10)\nm = 10 slots\n\nα = 9/10 = 0.9\n\nAnswer: 0.9\n\nNote for GATE: Load factor > 1 is possible with chaining (unlike open addressing where max is 1). With α = 0.9, average chain length is 0.9, meaning most slots have 0 or 1 element. This is a common GATE question pattern — just count keys and divide by slots."
}

# EXAMPLE 3: Placement Interview Conversation
{
  "messages": [
    {"role": "user", "content": "Interviewer (TCS Technical Round 1): Can you explain what a deadlock is and how to prevent it?"},
    {"role": "assistant", "content": "A deadlock is a situation where two or more processes are blocked forever, each waiting for a resource held by the other.\n\nClassic example: Process A holds Resource 1, needs Resource 2. Process B holds Resource 2, needs Resource 1. Both wait forever.\n\nFour necessary conditions (Coffman conditions):\n1. Mutual Exclusion: resource held by only one process\n2. Hold and Wait: process holds resource while waiting\n3. No Preemption: resource cannot be forcibly taken\n4. Circular Wait: circular chain of waiting\n\nPrevention (break any one condition):\n1. Mutual Exclusion: make resources sharable (not always possible)\n2. Hold and Wait: request all resources at once before starting\n3. Allow Preemption: take resources from waiting processes\n4. Circular Wait: order resources numerically, always request in order\n\nFor TCS interview: Say Banker's Algorithm for avoidance, and give the dining philosophers example if they ask for real-world scenario."},
    {"role": "user", "content": "What is the Banker's Algorithm?"},
    {"role": "assistant", "content": "Banker's Algorithm (deadlock avoidance, by Dijkstra):\n\nIdea: Before granting a resource request, check if doing so will leave the system in a 'safe state'.\n\nSafe state: There exists at least one sequence in which all processes can complete.\n\nData structures:\n- Available[j]: available instances of resource j\n- Max[i][j]: max demand of process i for resource j\n- Allocation[i][j]: currently allocated\n- Need[i][j] = Max[i][j] - Allocation[i][j]\n\nAlgorithm:\n1. If Need[i] ≤ Available: process i can proceed\n2. Simulate giving resources, add back after process completes\n3. If all processes can complete in some order: safe state\n4. Grant request only if safe state maintained\n\nFor TCS: They often ask you to trace through a small example. Practice with 3-4 processes and 2-3 resources."}
  ]
}

# EXAMPLE 4: Viva Question
{
  "instruction": "Viva question for DS Lab: What is the time complexity of inserting at the beginning of a singly linked list?",
  "output": "Time complexity: O(1) — constant time.\n\nReason: To insert at beginning:\n1. Create new node: O(1)\n2. Point new node's next to current head: O(1)\n3. Update head to new node: O(1)\n\nNo traversal needed — we always have direct access to head.\n\nContrast with inserting at end: O(n) because we must traverse to find last node (unless we maintain a tail pointer, then O(1)).\n\nCommon follow-up: 'What about doubly linked list?' — Same answer, O(1), just need to update prev pointer too."
}

# EXAMPLE 5: FYP Guide
{
  "instruction": "Suggest a final year project on AI/ML for a CSE student at an Indian university with intermediate Python skills and access to Google Colab.",
  "output": "## Recommended Project: Smart Attendance System with Face Recognition\n\n### Why This Project\n- Relevant to Indian college context\n- Achievable in one semester\n- Good for placements (Computer Vision + ML + Web Dev)\n- Free tools available (Google Colab, OpenCV)\n\n### Tech Stack\n- Frontend: HTML/CSS/JavaScript\n- Backend: Flask (Python)\n- ML: OpenCV + face_recognition library\n- Database: SQLite or MySQL\n- Deployment: Heroku (free tier) or college server\n\n### Modules\n1. Face Registration: capture and encode student faces\n2. Real-time Detection: webcam face recognition\n3. Attendance Marking: automatic DB update\n4. Report Generation: CSV/PDF export\n5. Admin Dashboard: web interface\n\n### 16-Week Timeline\nWeek 1-2: Literature review, requirement analysis\nWeek 3-4: Dataset collection, face detection basics\nWeek 5-6: Model training and testing\nWeek 7-8: Backend development (Flask API)\nWeek 9-10: Frontend development\nWeek 11-12: Integration testing\nWeek 13-14: Deployment and documentation\nWeek 15-16: Report writing and presentation\n\n### Expected Accuracy\n95%+ in good lighting conditions\n\n### Potential Extensions (for distinction)\n- Liveness detection (prevent photo spoofing)\n- Mobile app (Flutter)\n- Multiple camera support\n\n### Resources\n- face_recognition library: pip install face-recognition\n- OpenCV: pip install opencv-python\n- Tutorial: realpython.com/face-recognition-with-python"
}
```

---

## STAGE 4 — PREFERENCE OPTIMISATION (Optional)

### When to Do This

ONLY after Stage 2 + Stage 3 are stable and working.
Preference optimisation on a weak base model makes things WORSE.

### What It Does

SFT teaches the model to answer.
Preference optimisation teaches the model to give BETTER answers.

Example:
```
Question: "Explain recursion"

Answer A (bad): "Recursion is when a function calls itself."
Answer B (good): "Recursion is when a function calls itself to solve
  a smaller version of the same problem. Example:
  def factorial(n): return 1 if n==0 else n*factorial(n-1)
  Think of it like Russian dolls — each doll contains a smaller
  version of itself."

DPO trains model to prefer Answer B over Answer A.
```

### Datasets

```
UltraFeedback-binarized
  HuggingFace: HuggingFaceH4/ultrafeedback_binarized
  64K prompts, GPT-4 ranked responses
  Format: chosen (better) vs rejected (worse) pairs

Tulu 3 Preference
  HuggingFace: allenai/tulu-3-pref-mixture
  State of the art preference data

Magpie-Pro-DPO
  HuggingFace: magpie-align/Magpie-Pro-DPO-200K-v0.1
  200K DPO pairs from Magpie

YOUR OWN (most valuable):
  Create ranked engineering responses
  Ask 5 students to rate answers to GATE questions
  Use ratings as chosen/rejected pairs
  This produces domain-specific alignment
  No dataset has this for Indian engineering
```

---

## STAGE 5 — EVALUATION

### Why Evaluation Is the Research Stage

Running experiments without measuring is not research.
Every result in your paper comes from Stage 5.

### Required Measurements Every Training Run

```python
# evaluation/metrics.py

REQUIRED_METRICS = {
    # Training health
    "training_loss":       "curve over steps",
    "validation_loss":     "curve over steps",
    "perplexity":          "exp(val_loss) — lower is better",
    "tokens_per_sec":      "throughput measurement",
    "gpu_memory_mb":       "peak usage",
    "flops_per_step":      "computational cost",
    
    # Gradient health (critical for IVERI)
    "mamba2_grad_norm":    "should stay 0.1 - 5.0",
    "titans_grad_norm":    "should be ~10x lower than mamba2",
    "entropy_distribution":"should span full 0.0-1.0 range",
    "moe_expert_usage":    "all 4 experts should be used",
    "mor_depth_dist":      "should be non-uniform",
}
```

### Ablation Studies (MANDATORY for Publication)

```
ABLATION 1: Entropy routing vs learned routing
  Model A: IVERI full (entropy-conditioned MoE)
  Model B: IVERI with standard learned linear router
  Same everything else. Same params. Same data. Same steps.
  Question: does entropy routing help or hurt?
  This is REQUIRED by IEEE reviewers
  Results go in paper Table 2

ABLATION 2: FLOP-matched comparison
  IVERI 10M vs Transformer 10M vs Mamba-only 10M
  Same compute budget (FLOPs), not same params
  Track: validation loss per FLOP
  Question: does IVERI get better results per compute?
  This is the core efficiency claim
  Results go in paper Table 1 (most important table)

ABLATION 3: Titans vs no-Titans
  IVERI with Titans vs IVERI with static memory
  Long-context test: give fact at position 1,
  test recall at position 2000, 5000, 10000
  Results go in paper Table 3

ABLATION 4: MoR vs uniform depth
  IVERI with adaptive depth vs IVERI with depth=1 always
  Same params (depth 1 = simpler model)
  Does adaptive depth actually help?

ABLATION 5: BLT vs standard tokenizer
  IVERI with BLT vs IVERI with BPE tokenizer (baseline)
  Measure: tokens/patches processed per second
  Measure: quality on same validation set
  Results go in paper Table 4
```

### Benchmarks to Run

```
CODING BENCHMARKS
  HumanEval: 164 programming problems
    Metric: pass@1 (% problems solved correctly)
    Run with: bigcode/bigcode-evaluation-harness
    
  MBPP: 374 basic Python programming problems
    Metric: pass@1
    
  LiveCodeBench: contamination-free, updated regularly
    More reliable than HumanEval for 2026

REASONING BENCHMARKS
  GSM8K: 8,500 grade school math problems
    Metric: accuracy
    
  ARC-Challenge: science questions
    Metric: accuracy
    
  HellaSwag: commonsense completion
    Metric: accuracy

YOUR CUSTOM ENGINEERING BENCHMARK (most important for paper)
  Build a test set of exactly 200 questions:
    50 GATE CS questions (known answers, not in training)
    50 placement interview questions
    50 university exam questions
    50 coding problems
  
  Compare IVERI vs ChatGPT vs LLaMA 3.1 on same questions
  This is your paper's main result table
  This proves IVERI is better for your domain
  ChatGPT has no special training on these
  IVERI does
  IVERI should win
```

### Statistical Significance

```
Do not report a single number. Report with variance.
Run each experiment 3 times with different random seeds.
Report: mean ± standard deviation

Example:
  IVERI perplexity: 45.3 ± 0.8
  Transformer:      48.7 ± 1.2
  
  This is publishable.
  
  IVERI perplexity: 45.3 (one run)
  
  This is not publishable at IEEE/NeurIPS.
```

---

## STAGE 6 — SCALING

### Rule: Never Scale Blindly

Only move to next scale after current scale passes evaluation.

```
10M  → validate architecture trains stably
        validate Titans + Mamba2 coexist
        validate MoR depth is diverse
        validate MoE experts are all used
        PASS? → proceed to 35M

35M  → validate efficiency advantage appears
        FLOP comparison: IVERI vs transformer at 35M
        does gap grow from 10M? (it should)
        PASS? → proceed to 70M

70M  → validate coding benchmarks improve
        HumanEval at 70M vs HumanEval at 35M
        does coding quality scale properly?
        PASS? → proceed to 150M

150M → validate domain specialisation works
        Run your custom engineering benchmark
        Compare before and after Stage 3B fine-tuning
        PASS? → proceed to 300M

300M → full proof of concept
        All benchmarks
        Paper-ready results
        Patent filing point
```

### Hardware Per Scale

```
SCALE    HARDWARE               COST        NOTES
10M      RTX 3050 4GB (laptop)  Rs 0        seq_len=256, batch=8
35M      RTX 3050 + Kaggle T4   Rs 0        Kaggle free, 30GB
70M      Kaggle T4 x2           Rs 0        free, ~1 week training
150M     Colab Pro (A100 40GB)  ~Rs 800/mo  much faster
300M     Colab Pro A100         ~Rs 1500/mo or Vast.ai ~$0.30/hr

For scale 150M+:
  Enable FP16 mixed precision
  Enable gradient checkpointing
  Gradient accumulation steps = 8-16
  These fit 150M on A100 40GB
```

### Config Per Scale

```python
# configs/nano_10m.py
nano_config = IVERIConfig(
    hidden_dim=256, num_layers=6,
    batch_size=8, seq_len=256,
    gradient_accumulation=4
)  # ~36M (actual measured)

# configs/small_35m.py
small_config = IVERIConfig(
    hidden_dim=384, num_layers=8,
    batch_size=4, seq_len=512,
    gradient_accumulation=8
)

# configs/medium_70m.py
medium_config = IVERIConfig(
    hidden_dim=512, num_layers=10,
    batch_size=4, seq_len=512,
    gradient_accumulation=8
)

# configs/large_150m.py
large_config = IVERIConfig(
    hidden_dim=768, num_layers=12,
    batch_size=2, seq_len=1024,
    gradient_accumulation=16,
    mixed_precision="fp16",
    gradient_checkpointing=True
)

# configs/xlarge_300m.py
xlarge_config = IVERIConfig(
    hidden_dim=1024, num_layers=18,
    batch_size=2, seq_len=1024,
    gradient_accumulation=16,
    mixed_precision="fp16",
    gradient_checkpointing=True
)
```

---

## COMPLETE DATASET REFERENCE TABLE — JUNE 2026

```
STAGE 1 — FOUNDATION PRETRAINING
─────────────────────────────────────────────────────────────────────
Dataset              HuggingFace ID                    Size   Priority
─────────────────────────────────────────────────────────────────────
TinyStories          roneneldan/TinyStories             2GB    S-START
FineWeb-Edu          HuggingFaceFW/fineweb-edu           ~40GB  S
DCLM-Baseline        mlfoundations/dclm-baseline-1.0    ~30GB  S
Wikipedia (EN)       wikimedia/wikipedia                 20GB   A
FineMath             HuggingFaceFW/finemath              ~20GB  A
SlimPajama           cerebras/SlimPajama-627B            ~40GB  A
FineWeb-2            HuggingFaceFW/fineweb-2             ~20GB  B
The Stack v2 (Py)    bigcode/the-stack-v2 (Python)      ~50GB  A

STAGE 2 — INSTRUCTION TUNING (SFT)
─────────────────────────────────────────────────────────────────────
Magpie-Pro-1M        magpie-align/Magpie-Pro-1M-v0.1    ~2GB   S
Tulu 3 SFT           allenai/tulu-3-sft-mixture          ~1GB   S
OpenHermes 2.5       teknium/OpenHermes-2.5              ~1GB   A
WildChat             allenai/WildChat                    ~500MB A
Code-Feedback        m-a-p/Code-Feedback                 ~1GB   A
NuminaMath-CoT       AI-MO/NuminaMath-CoT                ~500MB A

STAGE 3A — CODING SPECIALISATION
─────────────────────────────────────────────────────────────────────
The Stack v2 (deep)  bigcode/the-stack-v2               ~50GB  S
Nemotron-Comp-v2     nvidia/Nemotron-SFT-Comp-v2         ~2GB   S
LeetCodeDataset      yunhui/LeetCodeDataset              ~500MB S
OpenCodeInstruct     nvidia/OpenCodeInstruct             ~1GB   A
Codeforces+R1        open-r1/codeforces                  ~1GB   A
CodeSearchNet        code_search_net                     ~2GB   A

STAGE 3B — IVERI PROPRIETARY (YOU BUILD)
─────────────────────────────────────────────────────────────────────
University Papers    Build from PDFs                     ~500MB S-UNIQUE
GATE CS 1991-2025    Build from official PDFs            ~100MB S-UNIQUE
GATE DA              Build from official PDFs            ~20MB  S-UNIQUE
Placement Q+A        Build from sources                  ~200MB S-UNIQUE
Subject Explanations Build yourself                      ~500MB S-UNIQUE
Viva Questions       Build from experience               ~100MB S-UNIQUE
Lab Procedures       Build from lab manuals              ~100MB S-UNIQUE
FYP Guides           Build yourself                      ~200MB S-UNIQUE
Debug Examples       Build from student code             ~100MB S-UNIQUE

STAGE 4 — PREFERENCE (Optional, Later)
─────────────────────────────────────────────────────────────────────
UltraFeedback        HuggingFaceH4/ultrafeedback_binarized ~200MB A
Tulu 3 Pref          allenai/tulu-3-pref-mixture          ~500MB A
Magpie-Pro-DPO       magpie-align/Magpie-Pro-DPO-200K     ~1GB   A

TOTAL: ~235GB (all free except your time for Stage 3B)
```

---

## DATA PIPELINE FOLDER STRUCTURE

```
iveri-core/
├── data/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── downloader.py        ← 0.1 download datasets
│   │   ├── license_checker.py   ← 0.2 verify licenses
│   │   ├── byte_encoder.py      ← 0.3 UTF-8 + byte encoding
│   │   ├── deduplication.py     ← 0.4 MinHash dedup
│   │   ├── language_detector.py ← 0.5 language detection
│   │   ├── quality_filter.py    ← 0.6 quality filtering
│   │   ├── pii_remover.py       ← 0.7 PII removal
│   │   ├── splitter.py          ← 0.8 train/val/test split
│   │   ├── versioning.py        ← 0.9 dataset versioning
│   │   ├── byte_counter.py      ← 0.10 byte counting
│   │   ├── mixer.py             ← 0.11 mixing weights
│   │   ├── sft_validator.py     ← 0.12 Q+A format validation
│   │   └── statistics.py        ← 0.13 dataset reports
│   │
│   ├── raw/                     ← downloaded datasets (gitignored)
│   │   ├── tinystories/
│   │   ├── fineweb_edu/
│   │   └── ...
│   │
│   ├── processed/               ← after pipeline (gitignored)
│   │   ├── stage1_pretrain/
│   │   ├── stage2_sft/
│   │   ├── stage3a_coding/
│   │   └── stage3b_proprietary/
│   │
│   ├── splits/                  ← train/val/test splits
│   │   ├── stage1_train.bin
│   │   ├── stage1_val.bin
│   │   └── ...
│   │
│   ├── proprietary/             ← your Indian engineering dataset
│   │   ├── university_papers/
│   │   ├── gate_questions/
│   │   ├── placement_qa/
│   │   ├── subject_explanations/
│   │   └── README.md
│   │
│   └── dataloader.py            ← the actual DataLoader class
```

---

## THE DATALOADER FOR IVERI (Byte-Level)

```python
# data/dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import os

class PretrainByteDataset(Dataset):
    """
    Dataset for Stage 1 Foundation Pretraining.
    Outputs raw bytes as input/target pairs.
    Next-byte prediction task.
    """
    def __init__(self, 
                  data_path: str,
                  seq_len: int = 256,
                  split: str = "train"):
        self.seq_len = seq_len
        
        # load as binary if pre-processed, or from text
        print(f"Loading {split} data from {data_path}...")
        ds = load_from_disk(data_path)
        
        # concatenate all text into one byte stream
        print("Encoding to bytes...")
        all_bytes = []
        for example in ds:
            text = example.get("text", example.get("content", ""))
            if text:
                all_bytes.extend(list(text.encode('utf-8')))
        
        self.data = np.array(all_bytes, dtype=np.uint8)
        print(f"Total bytes: {len(self.data):,} ({len(self.data)/1e9:.2f}GB)")
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()   # input
        y = torch.from_numpy(chunk[1:].copy()).long()    # target (next byte)
        return x, y


class SFTByteDataset(Dataset):
    """
    Dataset for Stage 2 Instruction Tuning.
    Formats Q+A pairs as byte sequences.
    Uses Alpaca or messages format.
    """
    PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )
    
    def __init__(self,
                  data_path: str,
                  seq_len: int = 512,
                  split: str = "train"):
        self.seq_len = seq_len
        
        ds = load_from_disk(data_path)
        self.samples = []
        
        for example in ds:
            text = self._format_sample(example)
            if text:
                self.samples.append(text.encode('utf-8'))
        
        print(f"SFT samples: {len(self.samples):,}")
    
    def _format_sample(self, example: dict) -> str:
        """Format sample to text string."""
        # messages format
        if "messages" in example:
            parts = []
            for msg in example["messages"]:
                role = msg["role"].capitalize()
                content = msg["content"]
                parts.append(f"### {role}:\n{content}")
            return "\n\n".join(parts)
        
        # alpaca format
        elif "instruction" in example and "output" in example:
            instruction = example["instruction"]
            inp = example.get("input", "")
            output = example["output"]
            
            if inp:
                instruction = f"{instruction}\n\nContext: {inp}"
            
            return self.PROMPT_TEMPLATE.format(
                instruction=instruction,
                output=output
            )
        
        return ""
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        byte_list = list(self.samples[idx])
        
        # truncate or pad to seq_len+1
        if len(byte_list) > self.seq_len + 1:
            byte_list = byte_list[:self.seq_len + 1]
        else:
            byte_list = byte_list + [0] * (self.seq_len + 1 - len(byte_list))
        
        byte_arr = np.array(byte_list, dtype=np.uint8)
        x = torch.from_numpy(byte_arr[:-1].copy()).long()
        y = torch.from_numpy(byte_arr[1:].copy()).long()
        return x, y


def get_pretrain_dataloader(data_path, batch_size=8, seq_len=256, 
                              num_workers=2, split="train"):
    dataset = PretrainByteDataset(data_path, seq_len=seq_len, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

def get_sft_dataloader(data_path, batch_size=4, seq_len=512,
                        num_workers=2, split="train"):
    dataset = SFTByteDataset(data_path, seq_len=seq_len, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
```

---

## RESEARCH LOG ADDITIONS FOR DATA

Add these fields to every experiment log entry:

```markdown
## Experiment [NUMBER]

### Data Configuration
- Stage: Stage 1 / 2 / 3A / 3B
- Datasets used: [list]
- Dataset versions: [version IDs from versioning.py]
- Mixing weights: [percentages]
- Total bytes in training set:
- Total bytes in validation set:
- Seq len used:
- SFT format validation: PASS/FAIL
- Dedup rate: X% removed

### Data Quality Observations
- Any anomalies in data?
- Entropy distribution of training batches:
- Any domain issues noticed?
```

---

## QUICK REFERENCE: WHAT YOUR SUPERVISOR SAID

```
1. "Provide question + answer too"
   → Stage 2 and Stage 3 MUST be in Q+A format
   → Every fine-tuning sample has instruction + output
   → Or messages with user + assistant turns
   → NOT raw paragraphs or notes

2. "Focus on specific domain, not general like ChatGPT"
   → Domain: CS/IT/AI/DS engineering education + coding
   → Stage 3B proprietary dataset is your main advantage
   → Compare IVERI vs ChatGPT on YOUR domain questions
   → IVERI should win because it's specifically trained

3. Implied: proper data engineering before training
   → Stage 0 pipeline (dedup, filter, validate, version)
   → Clean data = stable training
   → Reproducible experiments = publishable paper
```

---

## WHAT TO DO TODAY

```
□ Install Stage 0 dependencies:
  pip install datasets langdetect datasketch

□ Create data/ folder structure

□ Write downloader.py — download TinyStories first
  from datasets import load_dataset
  ds = load_dataset("roneneldan/TinyStories")
  Test it runs. Verify UTF-8 encoding works.

□ Write sft_validator.py — test with one Alpaca sample

□ Start collecting Stage 3B data (do this NOW):
  Search: "[your university] question papers CSE 2023 PDF"
  Download 10 papers as a start
  Manually convert 5 questions to Q+A format
  Test the format validator on them

□ Create proprietary/ folder
  Create README.md documenting sources and format
  This README is important for patent documentation
  (shows you built this original dataset)
```

---

*IVERI Data Pipeline Reference — June 2026*
*Document covers: Stage 0 through Stage 6*
*Key supervisor requirement: Q+A format for ALL fine-tuning*
*Key dataset updates: FineWeb-Edu > OpenWebText, Magpie > Alpaca*
*Unique advantage: Stage 3B proprietary Indian engineering dataset*
*Status: Ready to build Stage 0 pipeline*
