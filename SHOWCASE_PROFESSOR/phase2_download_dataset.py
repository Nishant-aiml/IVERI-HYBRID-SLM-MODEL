#!/usr/bin/env python3
"""
========================================================================================
IVERI-HYPERION: PHASE 2 — REAL ENGLISH DATASET PIPELINE (TinyStories)
========================================================================================
Downloads the TinyStories dataset (Microsoft Research) and prepares a clean
byte-level streaming dataloader for training IVERI-HYPERION on real English text.

TinyStories was specifically designed to let small models (<=33M params) learn
coherent, grammatical English stories with characters, actions, and concepts.

Outputs:
  - data/tinystories_train.txt  (Training split ~470MB)
  - data/tinystories_val.txt    (Validation split ~25MB)
  OR if download fails:
  - data/tinyshakespeare.txt    (Fallback: 1.1MB Shakespeare corpus)
========================================================================================
"""

import os, sys, json, urllib.request, zipfile, gzip
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_tinyshakespeare():
    """Fallback: Download Karpathy's tiny Shakespeare corpus (1.1MB)."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dest = DATA_DIR / "tinyshakespeare.txt"
    if dest.exists():
        print(f"  [OK] Already exists: {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  Downloading TinyShakespeare from {url}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  [OK] Downloaded: {dest} ({dest.stat().st_size:,} bytes)")
    return dest

def download_tinystories():
    """Primary: Use existing or download Microsoft TinyStories dataset."""
    train_dest = DATA_DIR / "tinystories_train.txt"
    val_dest = DATA_DIR / "tinystories_val.txt"
    
    if train_dest.exists() and train_dest.stat().st_size > 1_000_000:
        print(f"  [OK] Found existing TinyStories train corpus: {train_dest} ({train_dest.stat().st_size:,} bytes)")
        # If val_dest does not exist, split train_dest into 90% train and 10% val
        if not val_dest.exists() or val_dest.stat().st_size < 10_000:
            print("  Creating validation split from train corpus (10% slice)...")
            with open(train_dest, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2_000_000)  # read 2MB for val
            with open(val_dest, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [OK] Created validation file: {val_dest} ({val_dest.stat().st_size:,} bytes)")
        return train_dest, val_dest
    
    try:
        print(f"  Downloading TinyStories train split (may take a few minutes)...")
        urllib.request.urlretrieve(train_url, str(train_dest))
        print(f"  [OK] Train downloaded: {train_dest.stat().st_size:,} bytes")
        
        print(f"  Downloading TinyStories validation split...")
        urllib.request.urlretrieve(val_url, str(val_dest))
        print(f"  [OK] Val downloaded: {val_dest.stat().st_size:,} bytes")
        
        return train_dest, val_dest
    except Exception as e:
        print(f"  [WARN] TinyStories download failed: {e}")
        print(f"  Falling back to TinyShakespeare...")
        path = download_tinyshakespeare()
        return path, path

def create_byte_dataset_from_file(filepath, seq_len=512, max_sequences=None):
    """Load a text file and chunk into byte-level training sequences."""
    import torch
    
    print(f"\n  Loading text from: {filepath.name}")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    raw_bytes = list(text.encode("utf-8"))
    total_bytes = len(raw_bytes)
    print(f"  Total bytes in corpus: {total_bytes:,}")
    
    # Chunk into sequences of (seq_len + 1) for input/target pairs
    num_seqs = total_bytes // (seq_len + 1)
    if max_sequences:
        num_seqs = min(num_seqs, max_sequences)
    
    data = torch.tensor(raw_bytes[:num_seqs * (seq_len + 1)], dtype=torch.long)
    data = data.reshape(num_seqs, seq_len + 1)
    
    inputs = data[:, :-1]   # (N, seq_len)
    targets = data[:, 1:]   # (N, seq_len)
    
    print(f"  Created {num_seqs:,} training sequences of length {seq_len}")
    print(f"  Sample input (first 60 bytes decoded): \"{bytes(inputs[0, :60].tolist()).decode('utf-8', errors='replace')}\"")
    
    return inputs, targets


if __name__ == "__main__":
    print("=" * 80)
    print("  IVERI-HYPERION: Phase 2 — Real English Dataset Pipeline")
    print("=" * 80)
    
    print("\n[Step 1] Downloading Real English Text Corpus...")
    train_path, val_path = download_tinystories()
    
    print("\n[Step 2] Building Byte-Level Training Dataset...")
    import torch
    train_inputs, train_targets = create_byte_dataset_from_file(train_path, seq_len=512, max_sequences=5000)
    
    print("\n[Step 3] Building Byte-Level Validation Dataset...")
    val_inputs, val_targets = create_byte_dataset_from_file(val_path, seq_len=512, max_sequences=200)
    
    # Save processed tensors for fast loading
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)
    torch.save({"inputs": train_inputs, "targets": train_targets}, processed_dir / "train.pt")
    torch.save({"inputs": val_inputs, "targets": val_targets}, processed_dir / "val.pt")
    
    print(f"\n  [SAVED] Processed training tensors: {processed_dir / 'train.pt'}")
    print(f"  [SAVED] Processed validation tensors: {processed_dir / 'val.pt'}")
    
    print("\n" + "=" * 80)
    print("  [PHASE 2 COMPLETE] Real English Dataset Pipeline Ready!")
    print("=" * 80)
