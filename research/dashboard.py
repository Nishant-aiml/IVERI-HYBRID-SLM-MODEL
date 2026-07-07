# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research console dashboard rendering leaderboards, metrics, and progress states with connection closing."""

from __future__ import annotations

import logging
from typing import Any

from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)

# Rich import helper with graceful fallback
_HAS_RICH = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    pass


class ResearchDashboard:
    """Renders interactive leaderboards and scheduler pipelines in the console."""

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()
        if _HAS_RICH:
            self.console = Console()

    def get_leaderboard_data(self) -> list[dict[str, Any]]:
        """Query SQLite database to build model performance leaderboards."""
        conn = self.registry._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.experiment_id, e.purpose, e.random_seed, e.status, MIN(m.val_loss) as best_loss, MIN(m.perplexity) as best_ppl
                FROM experiments e
                LEFT JOIN metrics m ON e.experiment_id = m.experiment_id
                GROUP BY e.experiment_id
                ORDER BY best_ppl ASC
            """)
            rows = cursor.fetchall()
            return [
                {
                    "experiment_id": row[0],
                    "purpose": row[1],
                    "seed": row[2],
                    "status": row[3],
                    "best_loss": row[4] or 0.0,
                    "best_ppl": row[5] or 0.0,
                }
                for row in rows
            ]
        finally:
            conn.close()

    def render_dashboard(self) -> str:
        """Produce the console panel visualization string."""
        leaderboard = self.get_leaderboard_data()

        # Build ASCII Table representation
        table_lines = [
            "+----------------------+--------------------------------+--------+-----------+-----------+-----------+",
            "| Experiment ID        | Purpose                        | Seed   | Status    | Best Loss | Best PPL  |",
            "+----------------------+--------------------------------+--------+-----------+-----------+-----------+"
        ]
        for run in leaderboard:
            table_lines.append(
                f"| {run['experiment_id'][:20]:<20} | {run['purpose'][:30]:<30} | {run['seed']:<6} | {run['status']:<9} | {run['best_loss']:<9.4f} | {run['best_ppl']:<9.4f} |"
            )
        table_lines.append("+----------------------+--------------------------------+--------+-----------+-----------+-----------+")

        ascii_repr = "\n".join(table_lines)

        if _HAS_RICH:
            try:
                table = Table(title="IVERI CORE — Scientific Leaderboard")
                table.add_column("Experiment ID", style="cyan")
                table.add_column("Purpose", style="magenta")
                table.add_column("Seed", style="green", justify="right")
                table.add_column("Status", style="yellow")
                table.add_column("Best Loss", style="blue", justify="right")
                table.add_column("Best PPL", style="red", justify="right")

                for run in leaderboard:
                    table.add_row(
                        run["experiment_id"],
                        run["purpose"],
                        str(run["seed"]),
                        run["status"],
                        f"{run['best_loss']:.4f}",
                        f"{run['best_ppl']:.4f}"
                    )
                # Print panel
                self.console.print(Panel(table, title="IVERI CORE RESEARCH DASHBOARD", subtitle="Phase 3.6 Campaign Dashboard"))
            except Exception as e:
                logger.warning(f"Failed to render rich dashboard panel: {e}")

        return ascii_repr
