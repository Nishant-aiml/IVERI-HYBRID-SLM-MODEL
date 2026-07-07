# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Failure Replay Engine serializing and replaying training failures with complete RNG states."""

from __future__ import annotations

import json
import logging
import random
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)

# NumPy import wrapper
_HAS_NUMPY = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    pass


class FailureReplayEngine:
    """Manages failure payload serialization and deterministic replay execution sweeps."""

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def serialize_failure_payload(
        self,
        experiment_id: str,
        step: int,
        failure_type: str,
        error: Exception,
        input_tensor: torch.Tensor,
        config: IVERIConfig,
        payload_dir: str = "logs/failures/",
    ) -> Path:
        """Capture and serialize RNG states, input tensors, and error telemetry.

        Args:
            experiment_id: Active run ID.
            step: Training step index.
            failure_type: E.g., 'NaN', 'OOM', 'Expert_Collapse'.
            error: Caught Exception.
            input_tensor: Inputs batch tensor.
            config: Model config snapshot.
            payload_dir: Destination folder.

        Returns:
            Path: Path to serialized JSON payload.
        """
        payload_path = Path(payload_dir)
        payload_path.mkdir(parents=True, exist_ok=True)
        file_path = payload_path / f"failure_payload_{experiment_id}_{step}.json"

        # 1. Collect exact RNG states
        python_rng = random.getstate()
        
        numpy_rng_serializable = None
        if _HAS_NUMPY:
            numpy_rng = np.random.get_state()
            if numpy_rng is not None:
                numpy_rng_serializable = (
                    numpy_rng[0],
                    numpy_rng[1].tolist(),  # Convert ndarray to list
                    numpy_rng[2],
                    numpy_rng[3],
                    numpy_rng[4]
                )

        pytorch_rng = torch.get_rng_state().tolist()
        cuda_rng = []
        if torch.cuda.is_available():
            cuda_rng = [state.tolist() for state in torch.cuda.get_rng_state_all()]

        # Convert tensor inputs to list structures
        input_list = input_tensor.detach().cpu().tolist()

        payload = {
            "experiment_id": experiment_id,
            "step": step,
            "failure_type": failure_type,
            "error_message": str(error),
            "stack_trace": "".join(traceback.format_tb(error.__traceback__)),
            "config_dict": config.to_dict(),
            "input_shape": list(input_tensor.shape),
            "inputs": input_list,
            "rng": {
                "python_random": python_rng,
                "numpy_random": numpy_rng_serializable,
                "pytorch_cpu_rng": pytorch_rng,
                "pytorch_cuda_rng": cuda_rng,
            }
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)

        # Log entry in database
        self.registry.log_failure(
            experiment_id=experiment_id,
            step=step,
            failure_type=failure_type,
            error_message=str(error),
            stack_trace=payload["stack_trace"],
            rng_states=payload["rng"],
            payload_path=str(file_path),
        )

        logger.info(f"Successfully serialized failure payload dump to: {file_path}")
        return file_path

    def replay_failure(self, model: nn.Module, payload_path: str | Path) -> dict[str, Any]:
        """Restore exact RNG states and run a forward pass to deterministically replicate the failure.

        Args:
            model: PyTorch model instance.
            payload_path: Serialized JSON file.

        Returns:
            dict[str, Any]: Replay outcome metrics.
        """
        path = Path(payload_path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        rng = payload["rng"]

        # Restore RNG states
        py_rng = rng["python_random"]
        if py_rng:
            # Convert list back to tuple format for python random
            version = py_rng[0]
            state_tuple = tuple(py_rng[1])
            gauss = py_rng[2] if len(py_rng) > 2 else None
            random.setstate((version, state_tuple, gauss))

        if _HAS_NUMPY and rng["numpy_random"]:
            state_list = rng["numpy_random"]
            state_tuple = (
                state_list[0],
                np.array(state_list[1], dtype=np.uint32),
                state_list[2],
                state_list[3],
                state_list[4]
            )
            np.random.set_state(state_tuple)

        torch.set_rng_state(torch.ByteTensor(rng["pytorch_cpu_rng"]))
        if torch.cuda.is_available() and rng["pytorch_cuda_rng"]:
            # Restore CUDA RNG list
            torch.cuda.set_rng_state_all([torch.ByteTensor(state) for state in rng["pytorch_cuda_rng"]])

        # Prepare exact input tensor
        device = next(model.parameters()).device
        inputs = torch.tensor(payload["inputs"], dtype=torch.long, device=device)

        model.eval()
        caught_error = None
        tb_str = ""

        try:
            with torch.no_grad():
                _ = model(inputs)
        except Exception as e:
            caught_error = e
            tb_str = traceback.format_exc()

        return {
            "reproduced": caught_error is not None,
            "error_class": str(type(caught_error).__name__) if caught_error else "None",
            "error_message": str(caught_error) if caught_error else "",
            "stack_trace": tb_str,
        }
