# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research Artifacts Manager compiling reproducibility packages."""

from __future__ import annotations

import json
import logging
import os
import platform
import zipfile
import sys
from pathlib import Path
from typing import Any

import torch

from configs.base_config import IVERIConfig

logger = logging.getLogger(__name__)


class ResearchArtifactsManager:
    """Aggregates config logs, system metadata, and freezes packages into a reproducibility zip."""

    def __init__(self, config: IVERIConfig, output_dir: str = "reports/phase_3_5/artifacts/") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_system_info(self) -> dict[str, str]:
        """Gather host and environment details."""
        gpu_name = "none"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version.replace("\n", " "),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "none",
            "cudnn_version": str(torch.backends.cudnn.version()) or "none",
            "gpu_model": gpu_name,
            "cpu_architecture": platform.machine(),
            "cpu_processor": platform.processor(),
        }

    def _get_pip_freeze(self) -> list[str]:
        """Fetch list of active installed Python packages."""
        try:
            import importlib.metadata
            packages = sorted([
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions()
                if dist.metadata['Name']
            ])
            return packages
        except Exception:
            return ["importlib.metadata unavailable"]

    def export_reproducibility_package(
        self,
        experiment_metrics: dict[str, Any] | None = None,
        fig_paths: list[Path] | None = None,
    ) -> Path:
        """Collect all run logs, config settings, and environment specs into a reproducibility zip.

        Args:
            experiment_metrics: Dictionary of metric variables.
            fig_paths: List of generated figure Paths.

        Returns:
            Path: Path to the generated ZIP file.
        """
        manifest_path = self.output_dir / "reproducibility_manifest.json"
        
        # Compile metadata
        manifest_data = {
            "system_info": self._get_system_info(),
            "pip_freeze": self._get_pip_freeze(),
            "config_snapshot": self.config.to_dict(),
            "metrics_logs": experiment_metrics or {},
            "timestamp": time.time(),
        }

        # Write manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        zip_path = self.output_dir / "reproducibility_package.zip"
        
        # Write zip package
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.write(manifest_path, arcname="reproducibility_manifest.json")
            
            # Add config file if it exists locally
            config_file = Path("configs/base_config.py")
            if config_file.exists():
                zipf.write(config_file, arcname="base_config.py")

            # Add figures if provided
            if fig_paths:
                for path in fig_paths:
                    if path.exists():
                        zipf.write(path, arcname=f"figures/{path.name}")

        # Clean up temporary manifest file
        if manifest_path.exists():
            manifest_path.unlink()

        logger.info(f"Successfully packaged reproducibility archive: {zip_path}")
        return zip_path
import time
