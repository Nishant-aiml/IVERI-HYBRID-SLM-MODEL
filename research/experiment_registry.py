# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed Relational Experiment Registry with strict connection closing to prevent locks."""

from __future__ import annotations

import logging
import sqlite3
import json
from pathlib import Path
from typing import Any

from research.registry_integrity import (
    RegistryIntegrityError,
    assert_benchmark_exists,
    assert_benchmark_run_exists,
    assert_experiment_exists,
    assert_no_duplicate_experiment,
    audit_write,
    ensure_integrity_indexes,
    guard_benchmark_score_overwrite,
    guard_metric_provenance_overwrite,
    validate_experiment_id,
    validate_experiment_status,
    validate_provenance_label,
    validate_schema,
    validate_status_transition,
)

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    """Manages historical experiments registry using a local SQLite database."""

    def __init__(self, db_path: str = "research/experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initialize all relational database tables."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id TEXT PRIMARY KEY,
                        purpose TEXT,
                        hypothesis TEXT,
                        config_hash TEXT,
                        git_sha TEXT,
                        git_branch TEXT,
                        random_seed INTEGER,
                        tags TEXT,
                        provenance_label TEXT DEFAULT 'UNKNOWN',
                        status TEXT,
                        timestamp REAL,
                        version INTEGER DEFAULT 1
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        experiment_id TEXT,
                        step INTEGER,
                        train_loss REAL,
                        val_loss REAL,
                        perplexity REAL,
                        provenance_label TEXT DEFAULT 'UNKNOWN',
                        accuracy REAL,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS hardware (
                        experiment_id TEXT,
                        cpu_utilization REAL,
                        gpu_utilization REAL,
                        ram_peak_mb REAL,
                        vram_peak_mb REAL,
                        average_wattage REAL,
                        energy_joules REAL,
                        estimated_cost_usd REAL,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        dataset_id TEXT PRIMARY KEY,
                        name TEXT,
                        hash TEXT,
                        path TEXT,
                        license TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        step INTEGER,
                        path TEXT,
                        hash TEXT,
                        parameters_count INTEGER,
                        is_golden INTEGER DEFAULT 0,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS failures (
                        experiment_id TEXT,
                        step INTEGER,
                        failure_type TEXT,
                        error_message TEXT,
                        stack_trace TEXT,
                        rng_states TEXT,
                        payload_path TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        name TEXT,
                        path TEXT,
                        hash TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        experiment_id TEXT,
                        timestamp REAL,
                        author TEXT,
                        content TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_assets (
                        asset_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        type TEXT, -- 'table' or 'figure'
                        caption TEXT,
                        latex_label TEXT,
                        file_path TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benchmark_registry (
                        benchmark_id TEXT PRIMARY KEY,
                        name TEXT,
                        version TEXT,
                        source TEXT,
                        dataset_revision TEXT,
                        prompt_suite_version TEXT,
                        hash_sha256 TEXT,
                        num_prompts INTEGER,
                        evaluation_parameters TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        run_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        benchmark_id TEXT,
                        step INTEGER,
                        score REAL,
                        provenance_label TEXT DEFAULT 'UNKNOWN',
                        timestamp REAL,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
                        FOREIGN KEY(benchmark_id) REFERENCES benchmark_registry(benchmark_id)
                    )
                """)
                self._ensure_column(cursor, "experiments", "provenance_label", "TEXT DEFAULT 'UNKNOWN'")
                self._ensure_column(cursor, "metrics", "provenance_label", "TEXT DEFAULT 'UNKNOWN'")
                self._ensure_column(cursor, "benchmark_runs", "provenance_label", "TEXT DEFAULT 'UNKNOWN'")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benchmark_integrity (
                        run_id TEXT PRIMARY KEY,
                        prompt_hash_ok INTEGER,
                        template_hash_ok INTEGER,
                        system_prompt_hash_ok INTEGER,
                        fewshot_hash_ok INTEGER,
                        generation_params_hash_ok INTEGER,
                        dataset_hash_ok INTEGER,
                        reproducibility_ok INTEGER,
                        integrity_report_path TEXT,
                        FOREIGN KEY(run_id) REFERENCES benchmark_runs(run_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benchmark_artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT,
                        name TEXT,
                        path TEXT,
                        hash TEXT,
                        FOREIGN KEY(run_id) REFERENCES benchmark_runs(run_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS publication_runs (
                        pub_run_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        compiled_reports TEXT,
                        timestamp REAL,
                        directory_hash TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS release_manifests (
                        release_id TEXT PRIMARY KEY,
                        experiment_id TEXT,
                        release_hash TEXT,
                        metadata_json TEXT,
                        environment_info TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS db_write_audit (
                        audit_id TEXT PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        record_key TEXT NOT NULL,
                        payload_json TEXT,
                        timestamp REAL NOT NULL
                    )
                """)
                ensure_integrity_indexes(cursor.connection)
                validate_schema(cursor.connection)
                conn.commit()
        finally:
            conn.close()

    def _ensure_column(self, cursor: sqlite3.Cursor, table: str, column: str, column_def: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        if column not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

    def register_experiment(
        self,
        experiment_id: str,
        purpose: str,
        hypothesis: str,
        config_hash: str,
        git_sha: str,
        git_branch: str,
        random_seed: int,
        tags: list[str],
        provenance_label: str = "UNKNOWN",
        status: str = "PENDING",
        timestamp: float | None = None,
        version: int = 1,
    ) -> None:
        """Register a new experiment run entry (duplicate Run UUIDs are rejected)."""
        import time

        validate_experiment_id(experiment_id)
        validate_provenance_label(provenance_label)
        validate_experiment_status(status)
        t = timestamp or time.time()
        tags_str = ",".join(tags)
        conn = self._get_connection()
        try:
            with conn:
                assert_no_duplicate_experiment(conn, experiment_id)
                conn.execute(
                    """
                    INSERT INTO experiments (experiment_id, purpose, hypothesis, config_hash, git_sha, git_branch, random_seed, tags, provenance_label, status, timestamp, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        purpose,
                        hypothesis,
                        config_hash,
                        git_sha,
                        git_branch,
                        random_seed,
                        tags_str,
                        provenance_label,
                        status,
                        t,
                        version,
                    ),
                )
                audit_write(
                    conn,
                    "experiments",
                    "INSERT",
                    experiment_id,
                    {
                        "experiment_id": experiment_id,
                        "status": status,
                        "provenance_label": provenance_label,
                    },
                )
        finally:
            conn.close()

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        validate_experiment_id(experiment_id)
        validate_experiment_status(status)
        conn = self._get_connection()
        try:
            with conn:
                row = conn.execute(
                    "SELECT status FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                if row is None:
                    raise RegistryIntegrityError(
                        f"Cannot update status: experiment '{experiment_id}' not found."
                    )
                validate_status_transition(row[0], status)
                conn.execute(
                    "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                    (status, experiment_id),
                )
                audit_write(
                    conn,
                    "experiments",
                    "UPDATE_STATUS",
                    experiment_id,
                    {"from_status": row[0], "to_status": status},
                )
        finally:
            conn.close()

    def update_experiment_provenance(self, experiment_id: str, provenance_label: str) -> None:
        validate_experiment_id(experiment_id)
        validate_provenance_label(provenance_label)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                row = conn.execute(
                    "SELECT provenance_label FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                current = str(row[0]) if row else "UNKNOWN"
                if current == "MEASURED" and provenance_label != "MEASURED":
                    raise RegistryIntegrityError(
                        f"Provenance downgrade blocked for '{experiment_id}': "
                        "MEASURED cannot become non-MEASURED."
                    )
                conn.execute(
                    "UPDATE experiments SET provenance_label = ? WHERE experiment_id = ?",
                    (provenance_label, experiment_id),
                )
                audit_write(
                    conn,
                    "experiments",
                    "UPDATE_PROVENANCE",
                    experiment_id,
                    {"from_label": current, "to_label": provenance_label},
                )
        finally:
            conn.close()

    def log_metrics(
        self,
        experiment_id: str,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
        accuracy: float,
        provenance_label: str = "UNKNOWN",
    ) -> None:
        validate_experiment_id(experiment_id)
        validate_provenance_label(provenance_label)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                guard_metric_provenance_overwrite(conn, experiment_id, step, provenance_label)
                existing = conn.execute(
                    """
                    SELECT provenance_label FROM metrics
                    WHERE experiment_id = ? AND step = ?
                    """,
                    (experiment_id, step),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing[0]).upper() == "MEASURED"
                        and str(provenance_label).upper() == "MEASURED"
                    ):
                        conn.execute(
                            """
                            UPDATE metrics
                            SET train_loss = ?, val_loss = ?, perplexity = ?, accuracy = ?, provenance_label = ?
                            WHERE experiment_id = ? AND step = ?
                            """,
                            (
                                train_loss,
                                val_loss,
                                perplexity,
                                accuracy,
                                provenance_label,
                                experiment_id,
                                step,
                            ),
                        )
                        op = "UPDATE"
                    else:
                        raise RegistryIntegrityError(
                            f"Duplicate metric row blocked for {experiment_id} step {step}."
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO metrics (experiment_id, step, train_loss, val_loss, perplexity, provenance_label, accuracy)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            experiment_id,
                            step,
                            train_loss,
                            val_loss,
                            perplexity,
                            provenance_label,
                            accuracy,
                        ),
                    )
                    op = "INSERT"
                audit_write(
                    conn,
                    "metrics",
                    op,
                    f"{experiment_id}:{step}",
                    {
                        "experiment_id": experiment_id,
                        "step": step,
                        "provenance_label": provenance_label,
                        "val_loss": val_loss,
                    },
                )
        finally:
            conn.close()

    def log_hardware(
        self,
        experiment_id: str,
        cpu: float,
        gpu: float,
        ram_mb: float,
        vram_mb: float,
        wattage: float,
        energy_j: float,
        cost_usd: float,
    ) -> None:
        validate_experiment_id(experiment_id)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT INTO hardware (experiment_id, cpu_utilization, gpu_utilization, ram_peak_mb, vram_peak_mb, average_wattage, energy_joules, estimated_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (experiment_id, cpu, gpu, ram_mb, vram_mb, wattage, energy_j, cost_usd),
                )
                audit_write(
                    conn,
                    "hardware",
                    "INSERT",
                    experiment_id,
                    {"experiment_id": experiment_id, "gpu_utilization": gpu},
                )
        finally:
            conn.close()

    def register_checkpoint(
        self,
        checkpoint_id: str,
        experiment_id: str,
        step: int,
        path: str,
        chk_hash: str,
        parameters_count: int,
        is_golden: bool = False,
    ) -> None:
        validate_experiment_id(experiment_id)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (checkpoint_id, experiment_id, step, path, hash, parameters_count, is_golden)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (checkpoint_id, experiment_id, step, path, chk_hash, parameters_count, 1 if is_golden else 0),
                )
                audit_write(
                    conn,
                    "checkpoints",
                    "UPSERT",
                    checkpoint_id,
                    {"experiment_id": experiment_id, "step": step},
                )
        finally:
            conn.close()

    def log_failure(
        self,
        experiment_id: str,
        step: int,
        failure_type: str,
        error_message: str,
        stack_trace: str,
        rng_states: dict[str, Any],
        payload_path: str,
    ) -> None:
        validate_experiment_id(experiment_id)
        rng_str = json.dumps(rng_states)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT INTO failures (experiment_id, step, failure_type, error_message, stack_trace, rng_states, payload_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (experiment_id, step, failure_type, error_message, stack_trace, rng_str, payload_path),
                )
                audit_write(
                    conn,
                    "failures",
                    "INSERT",
                    f"{experiment_id}:{step}",
                    {"experiment_id": experiment_id, "failure_type": failure_type},
                )
        finally:
            conn.close()

    def add_note(self, experiment_id: str, author: str, content: str) -> None:
        import time

        validate_experiment_id(experiment_id)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    "INSERT INTO notes (experiment_id, timestamp, author, content) VALUES (?, ?, ?, ?)",
                    (experiment_id, time.time(), author, content),
                )
                audit_write(
                    conn,
                    "notes",
                    "INSERT",
                    experiment_id,
                    {"experiment_id": experiment_id, "author": author},
                )
        finally:
            conn.close()

    def register_paper_asset(
        self,
        asset_id: str,
        experiment_id: str,
        asset_type: str,
        caption: str,
        latex_label: str,
        file_path: str,
    ) -> None:
        validate_experiment_id(experiment_id)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_assets (asset_id, experiment_id, type, caption, latex_label, file_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (asset_id, experiment_id, asset_type, caption, latex_label, file_path),
                )
                audit_write(
                    conn,
                    "paper_assets",
                    "UPSERT",
                    asset_id,
                    {"experiment_id": experiment_id, "type": asset_type},
                )
        finally:
            conn.close()

    def get_experiments_by_tag(self, tag: str) -> list[dict[str, Any]]:
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE tags LIKE ?", (f"%{tag}%",))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def register_benchmark(
        self,
        benchmark_id: str,
        name: str,
        version: str,
        source: str,
        dataset_revision: str,
        prompt_suite_version: str,
        hash_sha256: str,
        num_prompts: int,
        evaluation_parameters: dict[str, Any],
    ) -> None:
        params_str = json.dumps(evaluation_parameters)
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO benchmark_registry (benchmark_id, name, version, source, dataset_revision, prompt_suite_version, hash_sha256, num_prompts, evaluation_parameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (benchmark_id, name, version, source, dataset_revision, prompt_suite_version, hash_sha256, num_prompts, params_str),
                )
                audit_write(
                    conn,
                    "benchmark_registry",
                    "UPSERT",
                    benchmark_id,
                    {"benchmark_id": benchmark_id, "version": version},
                )
        finally:
            conn.close()

    def log_benchmark_run(
        self,
        run_id: str,
        experiment_id: str,
        benchmark_id: str,
        step: int,
        score: float,
        provenance_label: str = "UNKNOWN",
    ) -> None:
        import time

        validate_experiment_id(experiment_id)
        validate_provenance_label(provenance_label)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                assert_benchmark_exists(conn, benchmark_id)
                guard_benchmark_score_overwrite(conn, run_id, provenance_label)
                existing = conn.execute(
                    "SELECT provenance_label FROM benchmark_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing[0]).upper() == "MEASURED"
                        and str(provenance_label).upper() == "MEASURED"
                    ):
                        conn.execute(
                            """
                            UPDATE benchmark_runs
                            SET experiment_id = ?, benchmark_id = ?, step = ?, score = ?, provenance_label = ?, timestamp = ?
                            WHERE run_id = ?
                            """,
                            (
                                experiment_id,
                                benchmark_id,
                                step,
                                score,
                                provenance_label,
                                time.time(),
                                run_id,
                            ),
                        )
                        op = "UPDATE"
                    else:
                        raise RegistryIntegrityError(
                            f"Duplicate benchmark run blocked for run_id '{run_id}'."
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO benchmark_runs (run_id, experiment_id, benchmark_id, step, score, provenance_label, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (run_id, experiment_id, benchmark_id, step, score, provenance_label, time.time()),
                    )
                    op = "INSERT"
                audit_write(
                    conn,
                    "benchmark_runs",
                    op,
                    run_id,
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "score": score,
                        "provenance_label": provenance_label,
                    },
                )
        finally:
            conn.close()

    def log_benchmark_integrity(
        self,
        run_id: str,
        prompt_hash_ok: bool,
        template_hash_ok: bool,
        system_prompt_hash_ok: bool,
        fewshot_hash_ok: bool,
        generation_params_hash_ok: bool,
        dataset_hash_ok: bool,
        reproducibility_ok: bool,
        integrity_report_path: str,
    ) -> None:
        conn = self._get_connection()
        try:
            with conn:
                assert_benchmark_run_exists(conn, run_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO benchmark_integrity (run_id, prompt_hash_ok, template_hash_ok, system_prompt_hash_ok, fewshot_hash_ok, generation_params_hash_ok, dataset_hash_ok, reproducibility_ok, integrity_report_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        1 if prompt_hash_ok else 0,
                        1 if template_hash_ok else 0,
                        1 if system_prompt_hash_ok else 0,
                        1 if fewshot_hash_ok else 0,
                        1 if generation_params_hash_ok else 0,
                        1 if dataset_hash_ok else 0,
                        1 if reproducibility_ok else 0,
                        integrity_report_path,
                    ),
                )
                audit_write(
                    conn,
                    "benchmark_integrity",
                    "UPSERT",
                    run_id,
                    {"run_id": run_id, "reproducibility_ok": reproducibility_ok},
                )
        finally:
            conn.close()

    def log_benchmark_artifact(
        self,
        artifact_id: str,
        run_id: str,
        name: str,
        path: str,
        hash_val: str,
    ) -> None:
        conn = self._get_connection()
        try:
            with conn:
                assert_benchmark_run_exists(conn, run_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO benchmark_artifacts (artifact_id, run_id, name, path, hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (artifact_id, run_id, name, path, hash_val),
                )
                audit_write(
                    conn,
                    "benchmark_artifacts",
                    "UPSERT",
                    artifact_id,
                    {"run_id": run_id, "name": name},
                )
        finally:
            conn.close()

    def log_publication_run(
        self,
        pub_run_id: str,
        experiment_id: str,
        compiled_reports: list[str],
        directory_hash: str,
    ) -> None:
        import time

        validate_experiment_id(experiment_id)
        reports_str = ",".join(compiled_reports)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO publication_runs (pub_run_id, experiment_id, compiled_reports, timestamp, directory_hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (pub_run_id, experiment_id, reports_str, time.time(), directory_hash),
                )
                audit_write(
                    conn,
                    "publication_runs",
                    "UPSERT",
                    pub_run_id,
                    {"experiment_id": experiment_id, "report_count": len(compiled_reports)},
                )
        finally:
            conn.close()

    def log_release_manifest(
        self,
        release_id: str,
        experiment_id: str,
        release_hash: str,
        metadata: dict[str, Any],
        env_info: dict[str, Any],
    ) -> None:
        metadata_str = json.dumps(metadata)
        env_str = json.dumps(env_info)
        validate_experiment_id(experiment_id)
        conn = self._get_connection()
        try:
            with conn:
                assert_experiment_exists(conn, experiment_id)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO release_manifests (release_id, experiment_id, release_hash, metadata_json, environment_info)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (release_id, experiment_id, release_hash, metadata_str, env_str),
                )
                audit_write(
                    conn,
                    "release_manifests",
                    "UPSERT",
                    release_id,
                    {"experiment_id": experiment_id, "release_hash": release_hash},
                )
        finally:
            conn.close()

    def count_write_audit_entries(self) -> int:
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT COUNT(*) FROM db_write_audit").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
