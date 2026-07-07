# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Experiment Scheduler managing priority queues, topological dependencies, and timelines with connection closing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class ExperimentScheduler:
    """Schedules, resolves dependencies, and logs timestamps of validation runs."""

    def __init__(
        self,
        registry: ExperimentRegistry | None = None,
        timeline_path: str = "reports/phase_3_6/Experiment_Timeline.md",
    ) -> None:
        self.registry = registry or ExperimentRegistry()
        self.timeline_path = Path(timeline_path)
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks: dict[str, dict[str, Any]] = {}
        self.dependencies: dict[str, set[str]] = {}

    def log_event(self, event_message: str) -> None:
        """Write timestamped event notes to the Markdown timeline logbook."""
        now_str = time.strftime("%H:%M:%S", time.localtime())
        date_str = time.strftime("%Y-%m-%d", time.localtime())

        # If file doesn't exist, create it with a header
        if not self.timeline_path.exists():
            with open(self.timeline_path, "w", encoding="utf-8") as f:
                f.write(f"# IVERI CORE — Experiment Execution Timeline\n\n**Date:** {date_str}\n\n| Time | Event Description |\n| --- | --- |\n")

        with open(self.timeline_path, "a", encoding="utf-8") as f:
            f.write(f"| {now_str} | {event_message} |\n")

    def add_task(
        self,
        task_id: str,
        execution_fn: Callable[[], Any],
        depends_on: list[str] | None = None,
        priority: int = 0,  # Higher is higher priority
    ) -> None:
        """Register a task with its callback function and dependencies list."""
        self.tasks[task_id] = {
            "execution_fn": execution_fn,
            "priority": priority,
            "status": "PENDING",
        }
        self.dependencies[task_id] = set(depends_on or [])

    def resolve_topological_order(self) -> list[str]:
        """Resolves task execution sequence matching dependency constraints.

        Uses Kahn's algorithm or equivalent sorting.
        """
        # Copy dependencies to modify
        in_degree = {u: len(self.dependencies[u]) for u in self.tasks}
        adj = {u: set() for u in self.tasks}
        for u, deps in self.dependencies.items():
            for v in deps:
                if v in adj:
                    adj[v].add(u)

        # Queue tasks with no incoming dependencies
        # Sort by priority desc for priority scheduling
        queue = [u for u in self.tasks if in_degree[u] == 0]
        queue.sort(key=lambda x: self.tasks[x]["priority"], reverse=True)

        ordered_list = []
        while queue:
            # Pop highest priority node
            curr = queue.pop(0)
            ordered_list.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            # Re-sort queue to maintain priority order
            queue.sort(key=lambda x: self.tasks[x]["priority"], reverse=True)

        if len(ordered_list) != len(self.tasks):
            # Circular dependency detected, fallback to priority only
            logger.warning("Circular dependency detected in scheduler graph. Falling back to priority sorting.")
            return sorted(self.tasks.keys(), key=lambda x: self.tasks[x]["priority"], reverse=True)

        return ordered_list

    def execute_campaign(self) -> dict[str, str]:
        """Execute scheduled campaign queue, skipping completed entries.

        Returns:
            dict[str, str]: Task ID to final execution status.
        """
        order = self.resolve_topological_order()
        self.log_event("Starting experiment campaign execution sweep.")
        results: dict[str, str] = {}

        for task_id in order:
            # Check SQLite status for interruption recovery
            conn = self.registry._get_connection()
            row = None
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM experiments WHERE experiment_id = ?", (task_id,))
                row = cursor.fetchone()
            finally:
                conn.close()

            if row and row[0] in ("SUCCESS", "COMPLETED"):
                logger.info(f"Task {task_id} already marked completed in registry database. Skipping.")
                self.log_event(f"Task {task_id} skipped (interruption recovery).")
                results[task_id] = "SKIPPED"
                continue

            logger.info(f"Executing scheduled task: {task_id}...")
            self.log_event(f"Task {task_id} execution started.")
            self.tasks[task_id]["status"] = "RUNNING"

            try:
                # Run callback function
                self.tasks[task_id]["execution_fn"]()
                self.tasks[task_id]["status"] = "COMPLETED"
                results[task_id] = "SUCCESS"
                self.log_event(f"Task {task_id} completed successfully.")
            except Exception as e:
                logger.error(f"Task {task_id} failed with error: {e}")
                self.tasks[task_id]["status"] = "FAILED"
                results[task_id] = "FAILED"
                self.log_event(f"Task {task_id} failed: {str(e)}")

        self.log_event("Finished experiment campaign execution sweep.")
        return results
