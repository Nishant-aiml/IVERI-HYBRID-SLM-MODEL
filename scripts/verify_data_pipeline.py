# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Verify SFT and Pretraining dataset loading and corruption rejection."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch

# Ensure root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import IVERIConfig
from training.pretraining_dataset import PretrainingDatasetLoader


def run_verification() -> bool:
    print("======================================================================")
    print("IVERI Data Pipeline Integration & Corruption Rejection Verification...")
    print("======================================================================")

    config = IVERIConfig()
    loader = PretrainingDatasetLoader(config)

    # 1. Load valid pretraining dataset
    print("[1/3] Loading valid tinystories pretraining dataset...")
    try:
        ds = loader.load("tinystories", split="train")
        print(f"Dataset successfully loaded. Number of items: {len(ds)}")
        x, y = ds[0]
        print(f"Sample shapes: x={x.shape}, y={y.shape}")
        assert x.shape == (512,), f"Expected shape (512,), got {x.shape}"
        assert y.shape == (512,), f"Expected shape (512,), got {y.shape}"
        print("Valid dataset load: OK")
    except Exception as e:
        print(f"Error loading valid dataset: {e}")
        return False

    # 2. Test Corruption Rejection: Modify content of a data file
    print("[2/3] Simulating content corruption in tinystories data folder...")
    data_pipeline = getattr(config, "data_pipeline", {})
    report_cfg = data_pipeline.get("report", {}) if isinstance(data_pipeline, dict) else getattr(data_pipeline, "report", {})
    processed_base = Path(getattr(report_cfg, "processed_data_dir", "data/processed") if not isinstance(report_cfg, dict) else report_cfg.get("processed_data_dir", "data/processed"))
    target_dir = processed_base / "stage1" / "tinystories"
    train_file = target_dir / "train.jsonl"
    
    # Back up train.jsonl
    backup_file = target_dir / "train.jsonl.bak"
    shutil.copyfile(train_file, backup_file)
    
    try:
        # Corrupt train.jsonl by appending garbage
        with open(train_file, "a", encoding="utf-8") as f:
            f.write("\nCorrupted line of text which changes the SHA-256 hash stamp!\n")
            
        print("Data file corrupted. Attempting to reload...")
        try:
            loader.load("tinystories", split="train")
            print("ERROR: Corrupted dataset was loaded successfully! Rejection failed.")
            return False
        except ValueError as val_err:
            print(f"Rejection success! Caught expected validation error: {val_err}")
            assert "hash mismatch" in str(val_err) or "does not match" in str(val_err)
    finally:
        # Restore backup
        shutil.copyfile(backup_file, train_file)
        os.remove(backup_file)

    # 3. Test Stage Mismatch: Modify stage in VERSION.json
    print("[3/3] Simulating stage mismatch in VERSION.json...")
    version_file = target_dir / "VERSION.json"
    with open(version_file, encoding="utf-8") as f:
        version_data = json.load(f)
        
    old_stage = version_data["stage"]
    version_data["stage"] = "2"  # Stage 2 instead of Stage 1
    
    backup_version = target_dir / "VERSION.json.bak"
    shutil.copyfile(version_file, backup_version)
    
    try:
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=4)
            
        print("VERSION.json modified with incorrect stage. Attempting to reload...")
        try:
            loader.load("tinystories", split="train")
            print("ERROR: Dataset with stage mismatch was loaded successfully! Rejection failed.")
            return False
        except ValueError as stage_err:
            print(f"Rejection success! Caught expected stage validation error: {stage_err}")
            assert "stage mismatch" in str(stage_err)
    finally:
        # Restore backup
        shutil.copyfile(backup_version, version_file)
        os.remove(backup_version)

    print("======================================================================")
    print("IVERI Data Pipeline: ALL INTEGRATION & CORRUPTION CHECKS PASSED.")
    print("======================================================================")
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
