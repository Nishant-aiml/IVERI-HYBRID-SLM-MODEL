# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable experiment registry for IVERI CORE.

Automatically logs every training run, benchmark, pilot, and ablation
into `reports/experiment_registry.json` for IEEE paper generation,
patent support, and experiment reproducibility.

Tracks full environment fingerprints, dataset SHA256 hashes, checkpoint
integrity, and random seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _get_git_sha() -> str:
    """Retrieve current Git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_file_sha256(path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file if it exists."""
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def get_environment_fingerprint() -> Dict[str, Any]:
    """Capture full hardware and software environment metadata."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    cudnn_ver = str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "None"

    gpu_driver = "Unknown"
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            gpu_driver = res.stdout.strip()
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "gpu_driver": gpu_driver,
        "gpu_name": gpu_name,
        "os": f"{platform.system()} {platform.release()}",
        "vram_gb": vram_gb,
    }


def register_experiment(
    experiment_id: str,
    stage: str,
    dataset: str,
    tokens: int,
    batch_size: int,
    seq_len: int,
    precision: str,
    grad_scaler: bool,
    seed: int = 42,
    optimizer: str = "AdamW",
    lr: float = 1e-3,
    wall_clock_sec: float = 0.0,
    peak_vram_mb: float = 0.0,
    tok_per_sec: float = 0.0,
    initial_loss: Optional[float] = None,
    final_loss: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    vocab_size: int = 259,
    train_val_split: str = "90/10",
    status: str = "completed",
    notes: str = "",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Appends an experiment record to reports/experiment_registry.json.

    Returns the complete registered record.
    """
    root_dir = Path(__file__).resolve().parent.parent
    registry_file = root_dir / "reports" / "experiment_registry.json"
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    git_sha = _get_git_sha()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    env_fingerprint = get_environment_fingerprint()

    # Checkpoint linkage
    ckpt_sha256 = None
    ckpt_size_mb = None
    if checkpoint_path:
        cp = root_dir / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
        ckpt_sha256 = _get_file_sha256(cp)
        ckpt_size_mb = round(cp.stat().st_size / (1024**2), 2) if cp.exists() else None

    # Dataset hash
    ds_hash = None
    if dataset_path:
        dp = root_dir / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)
        ds_hash = _get_file_sha256(dp)

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "stage": stage,
        "status": status,
        "notes": notes,
        "environment": env_fingerprint,
        "randomness_seeds": {
            "python_seed": seed,
            "numpy_seed": seed,
            "torch_seed": seed,
            "cuda_seed": seed,
        },
        "hyperparameters": {
            "dataset": dataset,
            "tokens": tokens,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "optimizer": optimizer,
            "lr": lr,
            "precision": precision,
            "grad_scaler": grad_scaler,
            "vocab_size": vocab_size,
            "train_val_split": train_val_split,
        },
        "metrics": {
            "wall_clock_sec": round(wall_clock_sec, 2),
            "wall_clock_hours": round(wall_clock_sec / 3600.0, 3),
            "tok_per_sec": round(tok_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "initial_loss": round(initial_loss, 4) if initial_loss is not None else None,
            "final_loss": round(final_loss, 4) if final_loss is not None else None,
        },
        "dataset_linkage": {
            "dataset_name": dataset,
            "dataset_path": dataset_path,
            "dataset_sha256": ds_hash,
        },
        "checkpoint_linkage": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": ckpt_sha256,
            "checkpoint_size_mb": ckpt_size_mb,
        },
        "extra_metrics": extra_metrics or {},
    }

    # Load existing registry
    registry: List[Dict[str, Any]] = []
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                registry = data.get("experiments", [])
        except Exception:
            registry = []

    # Check if experiment_id already exists -> update or append
    updated = False
    for i, item in enumerate(registry):
        if item.get("experiment_id") == experiment_id:
            registry[i] = record
            updated = True
            break

    if not updated:
        registry.append(record)

    output_data = {
        "metadata": {
            "title": "IVERI CORE — Machine-Readable Experiment Registry",
            "last_updated": timestamp,
            "total_experiments": len(registry),
            "reproducibility_policy": "Full Environment Fingerprint + SHA256 Checkpoint & Dataset Linkage",
        },
        "experiments": registry,
    }

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    return record
