# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Campaign Lock module freezing configurations and code states to prevent contamination."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CampaignLock:
    """Freezes experiment states and asserts git/dataset consistency on paper runs."""

    def __init__(self, lock_file_path: str = "research/campaign_lock.json") -> None:
        self.lock_path = Path(lock_file_path)

    def is_locked(self) -> bool:
        """Check if a campaign lock is currently active."""
        return self.lock_path.exists()

    def acquire_lock(
        self,
        config_hash: str,
        git_sha: str,
        dataset_hashes: dict[str, str],
        checkpoint_hashes: dict[str, str],
    ) -> None:
        """Lock the campaign settings to prevent modification."""
        lock_data = {
            "config_hash": config_hash,
            "git_sha": git_sha,
            "dataset_hashes": dataset_hashes,
            "checkpoint_hashes": checkpoint_hashes,
        }
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)
        logger.info(f"Campaign locked successfully. Frozen parameters recorded at: {self.lock_path}")

    def verify_lock_compliance(
        self,
        current_config_hash: str,
        current_git_sha: str,
        current_dataset_hashes: dict[str, str],
        current_checkpoint_hashes: dict[str, str],
    ) -> tuple[bool, list[str]]:
        """Verify that current parameters match the locked state.

        Returns:
            tuple[bool, list[str]]: (compliant, list of violations)
        """
        if not self.is_locked():
            return True, []

        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                locked = json.load(f)
        except Exception as e:
            return False, [f"Failed to read campaign lock file: {e}"]

        violations = []

        # Check configuration hash
        if locked.get("config_hash") != current_config_hash:
            violations.append(
                f"Configuration mismatch: Locked {locked.get('config_hash')}, current {current_config_hash}"
            )

        # Check git commit
        if locked.get("git_sha") != current_git_sha:
            violations.append(
                f"Codebase version mismatch: Locked git commit {locked.get('git_sha')}, current commit {current_git_sha}"
            )

        # Check dataset hashes
        locked_datasets = locked.get("dataset_hashes", {})
        for name, h in locked_datasets.items():
            if current_dataset_hashes.get(name) != h:
                violations.append(
                    f"Dataset '{name}' has changed: Locked hash {h}, current {current_dataset_hashes.get(name)}"
                )

        # Check checkpoint hashes
        locked_checkpoints = locked.get("checkpoint_hashes", {})
        for cid, h in locked_checkpoints.items():
            if current_checkpoint_hashes.get(cid) != h:
                violations.append(
                    f"Checkpoint '{cid}' has changed: Locked hash {h}, current {current_checkpoint_hashes.get(cid)}"
                )

        return len(violations) == 0, violations

    def release_lock(self) -> None:
        """Release the active campaign lock."""
        if self.is_locked():
            try:
                self.lock_path.unlink()
                logger.info("Campaign lock released successfully.")
            except Exception as e:
                logger.error(f"Failed to delete campaign lock file: {e}")
