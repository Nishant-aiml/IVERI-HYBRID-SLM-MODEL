# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Experiment logging and telemetry infrastructure for IVERI CORE.

Provides unified logging orchestration supporting Weights & Biases (online/offline),
local JSONL, CSV, and TensorBoard backends with a fail-safe fallback cascade:

    W&B → TensorBoard → CSV → JSONL

Training is never interrupted by a logging failure.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from utils.validation import get_gpu_memory_usage

# ── Optional backend imports ───────────────────────────────────────────────

try:
    import wandb as _wandb

    WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    _SummaryWriter = None  # type: ignore[assignment]
    TENSORBOARD_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────


def _sanitize(value: Any) -> Any:
    """Replace NaN / Inf floats with 0.0; leave other values intact."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        v = value.item()
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    return value


def _sanitize_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Sanitize all scalar values in a metric dictionary."""
    return {k: _sanitize(v) for k, v in metrics.items()}


def _git_info() -> dict[str, str]:
    """Return git commit hash and branch, or empty strings if not available."""
    result: dict[str, str] = {"git_commit": "", "git_branch": ""}
    try:
        result["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        result["git_branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass
    return result


def _system_info() -> dict[str, Any]:
    """Collect host hardware and software metadata."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }

    # GPU info
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_vram_mb"] = round(props.total_memory / 1024**2, 1)
    else:
        info["gpu_name"] = "N/A"
        info["gpu_vram_mb"] = 0.0

    # RAM
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        info["ram_total_mb"] = round(mem.total / 1024**2, 1)
    else:
        info["ram_total_mb"] = 0.0

    return info


# ── ExperimentLogger ───────────────────────────────────────────────────────


class ExperimentLogger:
    """Unified experiment logging orchestrator.

    Supports W&B, TensorBoard, CSV, and JSONL with automatic fallback:
    ``W&B → TensorBoard → CSV → JSONL``.

    Training is never interrupted by a logging failure — every backend
    operation is wrapped in a best-effort try/except.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize all active logging backends.

        Args:
            config: Full IVERI configuration object.
        """
        self.config = config
        lc = config.logging

        # Master kill-switch
        self.enabled: bool = lc.enabled and lc.mode != "disabled"
        if not self.enabled:
            self._init_null()
            return

        # Derive run identity
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name: str = lc.run_name or f"iveri-run-{timestamp}"
        self.run_id: str = lc.run_id or uuid.uuid4().hex[:8]

        # File paths
        self.save_dir = pathlib.Path(lc.save_dir)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            import tempfile

            self.save_dir = pathlib.Path(tempfile.mkdtemp(prefix="iveri-logs-"))
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / "metrics.csv"
        self.jsonl_path = self.save_dir / "metrics.jsonl"

        # Backend availability flags (updated below)
        self.use_wandb: bool = False
        self.use_tb: bool = False
        self.tb_writer: Any = None

        # --- W&B ---
        if WANDB_AVAILABLE and lc.mode in ("online", "offline"):
            self._init_wandb(config)

        # --- TensorBoard ---
        if TENSORBOARD_AVAILABLE and lc.tensorboard:
            self._init_tensorboard()

    # ── Private init helpers ───────────────────────────────────────────

    def _init_null(self) -> None:
        """Set all flags to no-op state."""
        self.run_name = ""
        self.run_id = ""
        self.save_dir = pathlib.Path("logs")
        self.csv_path = self.save_dir / "metrics.csv"
        self.jsonl_path = self.save_dir / "metrics.jsonl"
        self.use_wandb = False
        self.use_tb = False
        self.tb_writer = None

    def _init_wandb(self, config: IVERIConfig) -> None:
        """Attempt W&B initialisation; fall through on any error."""
        import tempfile

        lc = config.logging
        mode = "offline" if (lc.offline or lc.mode == "offline") else "online"
        wandb_dir = str(self.save_dir)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            wandb_dir = tempfile.mkdtemp(prefix="iveri-wandb-")
        try:
            _wandb.init(  # type: ignore[union-attr]
                project=lc.project,
                entity=lc.entity,
                name=self.run_name,
                id=self.run_id,
                config=config.to_dict(),
                mode=mode,
                dir=wandb_dir,
                resume=lc.resume,
                tags=lc.tags,
                notes=lc.notes,
            )
            self.use_wandb = True
        except (PermissionError, OSError, Exception) as exc:
            print(f"[ExperimentLogger] W&B init failed ({exc}); falling back to local backends.")
            self.use_wandb = False

    def _init_tensorboard(self) -> None:
        """Attempt TensorBoard SummaryWriter init; fall through on any error."""
        try:
            self.tb_writer = _SummaryWriter(  # type: ignore[call-arg]
                log_dir=str(self.save_dir / self.run_name)
            )
            self.use_tb = True
        except Exception as exc:
            print(f"[ExperimentLogger] TensorBoard init failed ({exc}); skipping.")

    # ── Public API ─────────────────────────────────────────────────────

    def log_experiment_metadata(self, seed: int = 42, dataset_version: str = "", dataset_hash: str = "") -> None:
        """Log immutable experiment metadata once at the start of a run.

        Captures architecture version, git info, system hardware, and the
        complete IVERIConfig snapshot.

        Args:
            seed: Global random seed for this run.
            dataset_version: Optional dataset version string.
            dataset_hash: Optional dataset content hash.
        """
        if not self.enabled:
            return

        sys_info = _system_info()
        git_info = _git_info()

        from core.constants import ARCHITECTURE_VERSION, IVERI_VERSION

        metadata: dict[str, Any] = {
            "meta/iveri_version": IVERI_VERSION,
            "meta/architecture_version": ARCHITECTURE_VERSION,
            "meta/run_name": self.run_name,
            "meta/run_id": self.run_id,
            "meta/timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "meta/random_seed": seed,
            "meta/dataset_version": dataset_version,
            "meta/dataset_hash": dataset_hash,
            "meta/git_commit": git_info["git_commit"],
            "meta/git_branch": git_info["git_branch"],
            **{f"system/{k}": v for k, v in sys_info.items()},
        }

        self.log(metadata)

    def log_hyperparameters(self) -> None:
        """Log the complete, recursively serialised IVERIConfig snapshot."""
        if not self.enabled:
            return
        flat = _flatten_dict(self.config.to_dict(), prefix="hparam")
        self.log(flat)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of metrics to every active backend.

        NaN / Inf values are replaced with 0.0 before dispatch.

        Args:
            metrics: Mapping of metric name to scalar value.
            step: Optional global training step index.
        """
        if not self.enabled:
            return

        cleaned = _sanitize_dict(metrics)
        cleaned["timestamp"] = time.time()
        if step is not None:
            cleaned["step"] = step

        # 1 — W&B
        if self.use_wandb:
            try:
                _wandb.log(cleaned, step=step)  # type: ignore[union-attr]
            except Exception as exc:
                print(f"[ExperimentLogger] W&B log failed ({exc}); continuing.")

        # 2 — TensorBoard
        if self.use_tb and self.tb_writer is not None and step is not None:
            try:
                for k, v in cleaned.items():
                    if isinstance(v, (int, float)) and k not in ("step", "timestamp"):
                        self.tb_writer.add_scalar(k, float(v), step)
            except Exception as exc:
                print(f"[ExperimentLogger] TensorBoard log failed ({exc}); continuing.")

        # 3 — CSV
        if self.config.logging.csv:
            try:
                self._write_csv(cleaned)
            except Exception as exc:
                print(f"[ExperimentLogger] CSV write failed ({exc}); continuing.")

        # 4 — JSONL
        if self.config.logging.json:
            try:
                self._write_jsonl(cleaned)
            except Exception as exc:
                print(f"[ExperimentLogger] JSONL write failed ({exc}); continuing.")

    def log_architecture_telemetry(
        self,
        model: nn.Module,
        step: int,
        telemetry_dict: dict[str, Any] | None = None,
    ) -> None:
        """Log architecture-specific telemetry and parameter/gradient statistics.

        Args:
            model: The IVERI model instance.
            step: Current global training step.
            telemetry_dict: Optional telemetry dict from the model forward pass.
        """
        if not self.enabled:
            return

        out: dict[str, Any] = {}

        # ── Architecture telemetry from forward pass ───────────────────
        if self.config.logging.telemetry_logging and telemetry_dict:
            for k, v in telemetry_dict.items():
                if isinstance(v, (int, float, bool, str)):
                    out[f"telemetry/{k}"] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    out[f"telemetry/{k}"] = v.item()

        # ── Gradient and parameter statistics ─────────────────────────
        if self.config.logging.gradient_logging:
            total_grad_sq = 0.0
            total_param_sq = 0.0
            total_params = 0
            trainable_params = 0
            frozen_params = 0
            grad_clip_count = 0

            for _name, p in model.named_parameters():
                n_params = p.numel()
                total_params += n_params

                if p.requires_grad:
                    trainable_params += n_params
                    p_norm = p.data.norm(2).item()
                    total_param_sq += p_norm**2
                    out[f"param_norm/{_name}"] = p_norm

                    if p.grad is not None:
                        g_norm = p.grad.data.norm(2).item()
                        g_max = p.grad.data.abs().max().item()
                        g_min = p.grad.data.abs().min().item()
                        out[f"grad_norm/{_name}"] = g_norm
                        out[f"grad_max/{_name}"] = g_max
                        out[f"grad_min/{_name}"] = g_min
                        total_grad_sq += g_norm**2
                        if g_norm > self.config.training.grad_clip:
                            grad_clip_count += 1
                else:
                    frozen_params += n_params

            out["param/total_count"] = total_params
            out["param/trainable_count"] = trainable_params
            out["param/frozen_count"] = frozen_params
            out["param/total_norm"] = float(math.sqrt(total_param_sq))
            out["grad/total_norm"] = float(math.sqrt(total_grad_sq))
            out["grad/clipping_count"] = grad_clip_count

        # ── Memory telemetry ──────────────────────────────────────────
        if self.config.logging.memory_logging:
            mem = get_gpu_memory_usage()
            out["memory/gpu_allocated_mb"] = mem.get("allocated", 0.0)
            out["memory/gpu_reserved_mb"] = mem.get("reserved", 0.0)
            out["memory/gpu_peak_mb"] = mem.get("peak", 0.0)

            if PSUTIL_AVAILABLE:
                proc = psutil.Process(os.getpid())
                rss = proc.memory_info().rss / 1024**2
                out["memory/cpu_ram_mb"] = round(rss, 1)

        self.log(out, step=step)

    def shutdown(self) -> None:
        """Finalise all active logging sessions."""
        if not self.enabled:
            return
        if self.use_wandb:
            try:
                _wandb.finish()  # type: ignore[union-attr]
            except Exception:
                pass
        if self.use_tb and self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass

    # ── Private file writers ───────────────────────────────────────────

    def _write_csv(self, metrics: dict[str, Any]) -> None:
        flat = {k: str(v) for k, v in metrics.items()}
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat)

    def _write_jsonl(self, metrics: dict[str, Any]) -> None:
        serial: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                serial[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                serial[k] = v.item()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serial) + "\n")


# ── Utility ────────────────────────────────────────────────────────────────


def _flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = "/") -> dict[str, Any]:
    """Recursively flatten a nested dictionary into dot-separated keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=full_key, sep=sep))
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[full_key] = v
    return out
