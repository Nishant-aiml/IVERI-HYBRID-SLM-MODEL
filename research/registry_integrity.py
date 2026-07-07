# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Database write guards, schema validation, and audit helpers for experiments.db (Phase 6.3.1B)."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

PROVENANCE_LABELS = frozenset({"MEASURED", "SYNTHETIC", "PILOT", "VERIFICATION", "UNKNOWN"})
EXPERIMENT_STATUSES = frozenset({"PENDING", "RUNNING", "COMPLETED", "FAILED", "SUCCESS"})
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "SUCCESS"})

# PENDING may not jump directly to COMPLETED/SUCCESS; FAILED may never recover.
ALLOWED_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"RUNNING", "FAILED"}),
    "RUNNING": frozenset({"COMPLETED", "SUCCESS", "FAILED"}),
    "COMPLETED": frozenset(),
    "SUCCESS": frozenset(),
    "FAILED": frozenset(),
}

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "experiments": (
        "experiment_id",
        "purpose",
        "hypothesis",
        "config_hash",
        "git_sha",
        "git_branch",
        "random_seed",
        "tags",
        "provenance_label",
        "status",
        "timestamp",
        "version",
    ),
    "metrics": (
        "experiment_id",
        "step",
        "train_loss",
        "val_loss",
        "perplexity",
        "provenance_label",
        "accuracy",
    ),
    "benchmark_runs": (
        "run_id",
        "experiment_id",
        "benchmark_id",
        "step",
        "score",
        "provenance_label",
        "timestamp",
    ),
    "benchmark_registry": (
        "benchmark_id",
        "name",
        "version",
        "source",
        "dataset_revision",
        "prompt_suite_version",
        "hash_sha256",
        "num_prompts",
        "evaluation_parameters",
    ),
    "checkpoints": (
        "checkpoint_id",
        "experiment_id",
        "step",
        "path",
        "hash",
        "parameters_count",
        "is_golden",
    ),
    "db_write_audit": (
        "audit_id",
        "table_name",
        "operation",
        "record_key",
        "payload_json",
        "timestamp",
    ),
}


class RegistryIntegrityError(Exception):
    """Raised when a registry write violates Phase 6.3.1B integrity rules."""


def validate_provenance_label(label: str) -> None:
    if label not in PROVENANCE_LABELS:
        raise RegistryIntegrityError(f"Invalid provenance_label '{label}'.")


def validate_experiment_status(status: str) -> None:
    if status not in EXPERIMENT_STATUSES:
        raise RegistryIntegrityError(f"Invalid experiment status '{status}'.")


def validate_status_transition(current: str | None, new: str) -> None:
    validate_experiment_status(new)
    if current is None:
        return
    current_norm = str(current).upper()
    new_norm = str(new).upper()
    if current_norm == new_norm:
        return
    if current_norm == "FAILED" and new_norm in {"COMPLETED", "SUCCESS", "RUNNING", "PENDING"}:
        raise RegistryIntegrityError(
            f"Status transition blocked: FAILED cannot become {new_norm}."
        )
    if current_norm == "PENDING" and new_norm in {"COMPLETED", "SUCCESS"}:
        raise RegistryIntegrityError(
            f"Status transition blocked: PENDING cannot become {new_norm} "
            "(must transition through RUNNING)."
        )
    allowed = ALLOWED_STATUS_TRANSITIONS.get(current_norm, frozenset())
    if new_norm not in allowed:
        raise RegistryIntegrityError(
            f"Status transition blocked: {current_norm} -> {new_norm} is not permitted."
        )


def validate_experiment_id(experiment_id: str) -> None:
    if not experiment_id or not str(experiment_id).strip():
        raise RegistryIntegrityError("experiment_id (Run UUID) must be non-empty.")


def assert_experiment_exists(conn: sqlite3.Connection, experiment_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise RegistryIntegrityError(
            f"Foreign-key validation failed: experiment '{experiment_id}' does not exist."
        )


def assert_benchmark_exists(conn: sqlite3.Connection, benchmark_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM benchmark_registry WHERE benchmark_id = ?",
        (benchmark_id,),
    ).fetchone()
    if row is None:
        raise RegistryIntegrityError(
            f"Foreign-key validation failed: benchmark '{benchmark_id}' does not exist."
        )


def assert_benchmark_run_exists(conn: sqlite3.Connection, run_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM benchmark_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise RegistryIntegrityError(
            f"Foreign-key validation failed: benchmark run '{run_id}' does not exist."
        )


def assert_no_duplicate_experiment(conn: sqlite3.Connection, experiment_id: str) -> None:
    row = conn.execute(
        "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if row is not None:
        raise RegistryIntegrityError(
            f"Duplicate Run UUID blocked: experiment_id '{experiment_id}' already exists."
        )


def guard_metric_provenance_overwrite(
    conn: sqlite3.Connection,
    experiment_id: str,
    step: int,
    new_label: str,
) -> None:
    """Reject writes that would replace MEASURED metrics with non-MEASURED values."""
    validate_provenance_label(new_label)
    row = conn.execute(
        """
        SELECT provenance_label FROM metrics
        WHERE experiment_id = ? AND step = ?
        """,
        (experiment_id, step),
    ).fetchone()
    if row is None:
        return
    existing = str(row[0]).upper()
    incoming = str(new_label).upper()
    if existing == "MEASURED" and incoming != "MEASURED":
        raise RegistryIntegrityError(
            f"Metric overwrite blocked: MEASURED row at {experiment_id} step {step} "
            f"cannot be replaced with {incoming}."
        )


def guard_benchmark_score_overwrite(
    conn: sqlite3.Connection,
    run_id: str,
    new_label: str,
) -> None:
    validate_provenance_label(new_label)
    row = conn.execute(
        "SELECT provenance_label FROM benchmark_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return
    existing = str(row[0]).upper()
    incoming = str(new_label).upper()
    if existing == "MEASURED" and incoming != "MEASURED":
        raise RegistryIntegrityError(
            f"Benchmark overwrite blocked: MEASURED run '{run_id}' "
            f"cannot be replaced with {incoming} provenance."
        )


def audit_write(
    conn: sqlite3.Connection,
    table_name: str,
    operation: str,
    record_key: str,
    payload: dict[str, Any],
) -> None:
    audit_id = f"{table_name}:{record_key}:{int(time.time() * 1000)}"
    conn.execute(
        """
        INSERT INTO db_write_audit (audit_id, table_name, operation, record_key, payload_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            audit_id,
            table_name,
            operation,
            record_key,
            json.dumps(payload, sort_keys=True, default=str),
            time.time(),
        ),
    )
    logger.debug("DB write audited: %s %s %s", operation, table_name, record_key)


def validate_schema(conn: sqlite3.Connection) -> None:
    """Verify required tables and columns exist."""
    for table, columns in REQUIRED_TABLES.items():
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None:
            raise RegistryIntegrityError(f"Schema validation failed: missing table '{table}'.")
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        present = {r[1] for r in info}
        missing = [c for c in columns if c not in present]
        if missing:
            raise RegistryIntegrityError(
                f"Schema validation failed: table '{table}' missing columns {missing}."
            )


def ensure_integrity_indexes(conn: sqlite3.Connection) -> None:
    """Create integrity indexes, deduplicating legacy metric rows if required."""
    conn.execute(
        """
        DELETE FROM metrics
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM metrics GROUP BY experiment_id, step
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_experiment_step
        ON metrics(experiment_id, step)
        """
    )
