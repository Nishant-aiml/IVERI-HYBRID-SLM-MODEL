# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Integrity Framework for IVERI CORE.

Locks prompt suites, datasets, template configurations, and checks environment-level reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from evaluation.contamination_checker import ContaminationChecker
from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class BenchmarkIntegrityFramework:
    """Audits evaluation prompt suites, templates, datasets, and environmental variables."""

    def __init__(self, data_dir: str | Path = "data/", db_path: str = "research/experiments.db") -> None:
        self.data_dir = Path(data_dir)
        self.registry = ExperimentRegistry(db_path)
        self.contamination_checker = ContaminationChecker(ngram_size=8, similarity_threshold=0.8)

    def compute_file_sha256(self, filepath: Path) -> str:
        """Compute the SHA256 of a file, reading in 1MB chunks to avoid memory issues."""
        if not filepath.exists() or not filepath.is_file():
            return "missing"
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.error("Failed to compute hash for %s: %s", filepath, e)
            return "error"

    def lock_dataset_revisions(self) -> dict[str, str]:
        """Locks all dataset binaries, manifests, config files, and tokenizers."""
        locks: dict[str, str] = {}
        target_files = [
            self.data_dir / "pretrain.bin",
            self.data_dir / "validation.bin",
            self.data_dir / "VERSION.json",
            self.data_dir / "dataset_manifest.json",
            Path("configs/base_config.py"),
            Path("data/preprocessing.py"),
            Path("data/dataloader.py"),
        ]

        for path in target_files:
            if path.exists():
                locks[path.as_posix()] = self.compute_file_sha256(path)
            else:
                locks[path.as_posix()] = "missing"

        return locks

    def _hash_callable(self, func: Callable[..., Any] | None) -> str:
        """Computes SHA256 of a function source code or representation."""
        if func is None:
            return "none"
        try:
            import inspect
            source = inspect.getsource(func)
            return hashlib.sha256(source.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.debug("Failed to get source of callable %s, falling back to repr: %s", func, e)
            return hashlib.sha256(str(func).encode("utf-8")).hexdigest()

    def hash_pipeline_assets(
        self,
        benchmark_name: str,
        prompts: list[dict[str, Any]],
        template_func: Callable[..., Any] | None,
        system_prompt: str,
        evaluation_config: dict[str, Any],
        few_shot: list[dict[str, Any]] | None,
        generation_params: dict[str, Any],
    ) -> dict[str, str]:
        """Hashes prompts, templates, system prompts, few-shot examples, and parameters."""
        prompt_str = json.dumps(prompts, sort_keys=True)
        few_shot_str = json.dumps(few_shot or [], sort_keys=True)
        eval_cfg_str = json.dumps(evaluation_config, sort_keys=True)
        gen_params_str = json.dumps(generation_params, sort_keys=True)

        return {
            "prompt_hash": hashlib.sha256(prompt_str.encode("utf-8")).hexdigest(),
            "template_hash": self._hash_callable(template_func),
            "system_prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "few_shot_hash": hashlib.sha256(few_shot_str.encode("utf-8")).hexdigest(),
            "evaluation_config_hash": hashlib.sha256(eval_cfg_str.encode("utf-8")).hexdigest(),
            "generation_params_hash": hashlib.sha256(gen_params_str.encode("utf-8")).hexdigest(),
        }

    def get_env_info(self) -> dict[str, Any]:
        """Collects environmental factors for reproducibility audits."""
        git_sha = "unknown"
        git_branch = "unknown"
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass

        pip_freeze = "unknown"
        try:
            pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass

        gpu_name = "CPU"
        cuda_driver = "none"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_driver = torch.version.cuda

        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gpu": gpu_name,
            "cuda_driver": cuda_driver,
            "git_sha": git_sha,
            "git_branch": git_branch,
            "pip_freeze": pip_freeze,
        }

    def run_contamination_check(
        self,
        benchmark_name: str,
        prompts: list[dict[str, Any]],
        data_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Runs the ContaminationChecker and yields verification results."""
        target_dir = data_dir or self.data_dir
        report = self.contamination_checker.check(prompts, target_dir)
        return {
            "benchmark_name": benchmark_name,
            "total_prompts": report.total_prompts,
            "contaminated_count": report.contaminated_count,
            "contamination_ratio": report.contamination_ratio,
            "clean": report.clean,
        }

    def audit_reproducibility(
        self,
        run_config: dict[str, Any],
        expected_env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Audits current environment specs against target/expected configurations."""
        current_env = self.get_env_info()
        reproducibility_ok = True
        mismatches: list[str] = []

        # Validate core settings
        cfg_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode("utf-8")).hexdigest()

        if expected_env:
            for key in ["pytorch_version", "numpy_version", "git_sha"]:
                curr = current_env.get(key)
                exp = expected_env.get(key)
                if exp and curr != exp:
                    reproducibility_ok = False
                    mismatches.append(f"Mismatch in {key}: expected '{exp}', got '{curr}'")

        return {
            "reproducibility_ok": reproducibility_ok,
            "config_hash": cfg_hash,
            "mismatches": mismatches,
            "current_env": current_env,
        }
