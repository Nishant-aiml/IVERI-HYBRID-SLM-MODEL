# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Supervised Fine-Tuning (SFT) Q+A dataset format validator.

Enforces strict structural constraints on instruction tuning data:
- Rejects placeholders ("TODO", "N/A", etc.)
- Enforces user/assistant alternation and role constraints
- Checks sequence lengths, character bounds, and UTF-8 encoding compliance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class SFTValidationReport:
    """Report detailing SFT validation results."""

    total: int
    valid_count: int
    invalid_count: int
    invalid_rate: float
    errors: list[dict[str, Any]] = field(default_factory=list)
    status: str = "PASS"  # PASS or FAIL
    validation_time: str = ""


class SFTValidator:
    """Validates structural correctness of SFT data samples."""

    PLACEHOLDER_OUTPUTS = {
        "todo",
        "placeholder",
        "...",
        "n/a",
        "tbd",
        "answer here",
        "[answer]",
        "",
        "none",
    }
    VALID_ROLES = {"user", "assistant", "system"}

    def validate_alpaca(self, sample: dict[str, Any]) -> tuple[bool, str]:
        """Validate Alpaca single-turn format (instruction/output)."""
        if "instruction" not in sample:
            return False, "missing 'instruction' field"
        if "output" not in sample:
            return False, "missing 'output' field"

        inst = sample["instruction"]
        out = sample["output"]

        # Type checks
        if not isinstance(inst, str):
            return False, f"'instruction' must be a string, got {type(inst).__name__}"
        if not isinstance(out, str):
            return False, f"'output' must be a string, got {type(out).__name__}"

        # Placeholder checks
        if out.strip().lower() in self.PLACEHOLDER_OUTPUTS:
            return False, f"output contains placeholder: '{out}'"

        # Length constraints
        if len(inst.strip()) < 5:
            return False, "instruction is too short (min 5 chars)"
        if len(out.strip()) < 10:
            return False, "output is too short (min 10 chars)"

        # Byte length validation (max 50,000 bytes per sample)
        combined_bytes = len(inst.encode("utf-8")) + len(out.encode("utf-8"))
        if combined_bytes > 50_000:
            return False, f"sample size {combined_bytes} bytes exceeds maximum limit (50,000)"

        return True, "OK"

    def validate_conversation(self, sample: dict[str, Any]) -> tuple[bool, str]:
        """Validate Multi-turn messages list format."""
        if "messages" not in sample:
            return False, "missing 'messages' field"

        messages = sample["messages"]
        if not isinstance(messages, list):
            return False, f"'messages' must be a list, got {type(messages).__name__}"

        if len(messages) < 2:
            return False, "conversation must have at least 2 messages"

        if len(messages) > 50:
            return False, "conversation exceeds maximum of 50 turns"

        # Track roles and validate order
        prev_role = None
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"message at index {i} must be a dictionary"

            if "role" not in msg or "content" not in msg:
                return False, f"message at index {i} is missing 'role' or 'content'"

            role = msg["role"]
            content = msg["content"]

            if role not in self.VALID_ROLES:
                return False, f"message at index {i} has invalid role: '{role}'"

            if not isinstance(content, str):
                return False, f"message at index {i} content must be a string"

            # Check for empty content
            if len(content.strip()) == 0:
                return False, f"message at index {i} has empty content"

            # Alternate role validation (ignoring system role at the beginning)
            if role == prev_role and role != "system":
                return False, f"message at index {i} has consecutive duplicate role: '{role}'"

            prev_role = role

        # Must start with user (or system then user)
        first_role = messages[0]["role"]
        if first_role == "system":
            if len(messages) < 3:
                return False, "system message must be followed by a user and assistant exchange"
            second_role = messages[1]["role"]
            if second_role != "user":
                return False, f"message following system role must be 'user', got '{second_role}'"
        elif first_role != "user":
            return False, f"first message role must be 'user' or 'system', got '{first_role}'"

        # Must end with assistant
        last_role = messages[-1]["role"]
        if last_role != "assistant":
            return False, f"conversation must end with 'assistant' role, got '{last_role}'"

        # Check for placeholders in assistant responses
        for i, msg in enumerate(messages):
            if (
                msg["role"] == "assistant"
                and msg["content"].strip().lower() in self.PLACEHOLDER_OUTPUTS
            ):
                return False, f"assistant response at index {i} is a placeholder"

        # Byte length validation (max 50,000 bytes)
        total_bytes = sum(len(msg["content"].encode("utf-8")) for msg in messages)
        if total_bytes > 50_000:
            return False, f"conversation size {total_bytes} bytes exceeds maximum limit (50,000)"

        return True, "OK"

    def validate_sample(self, sample: dict[str, Any]) -> tuple[bool, str]:
        """Auto-dispatch sample to appropriate validator."""
        if "messages" in sample:
            return self.validate_conversation(sample)
        elif "instruction" in sample:
            return self.validate_alpaca(sample)
        return False, "sample does not match 'messages' or 'alpaca' formats"

    def validate_dataset(
        self, dataset: list[dict[str, Any]], max_errors: int = 20
    ) -> SFTValidationReport:
        """Validate an entire list of SFT dataset samples."""
        valid = 0
        invalid = 0
        errors: list[dict[str, Any]] = []

        for i, sample in enumerate(dataset):
            ok, err = self.validate_sample(sample)
            if ok:
                valid += 1
            else:
                invalid += 1
                if len(errors) < max_errors:
                    errors.append(
                        {
                            "index": i,
                            "error": err,
                            "sample_preview": str(sample)[:100] + "...",
                        }
                    )

        status = "PASS" if invalid == 0 else "FAIL"
        total = len(dataset)
        return SFTValidationReport(
            total=total,
            valid_count=valid,
            invalid_count=invalid,
            invalid_rate=(invalid / total) if total > 0 else 0.0,
            errors=errors,
            status=status,
            validation_time=datetime.now().isoformat(),
        )

    def filter_valid(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter and return only valid SFT samples."""
        valid_samples = []
        for sample in dataset:
            ok, _ = self.validate_sample(sample)
            if ok:
                valid_samples.append(sample)
        return valid_samples
