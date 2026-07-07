# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for seeding publication-ready measured experiment data in tests."""

from __future__ import annotations

from research.experiment_registry import ExperimentRegistry

GIT_SHA = "a1b2c3d4e5f6789012345678abcdef9012345678"
CONFIG_HASH = "c4f8e2b19d0a63f7b5e1c9024d8a6f3e91bc0472"
CHECKPOINT_HASH = "d9a1f3e7b2c6049581e3d4f6a8b0c2e5f7a913b4c6"
RELEASE_HASH = "e1b2c3d4e5f6789012345678901234567890abcd"
BENCHMARK_REGISTRY_HASH = "f1e2d3c4b5a6978877665544433221100abcdef12"


def seed_measured_experiment(
    registry: ExperimentRegistry,
    experiment_id: str = "exp_run_1",
    *,
    hypothesis: str = "H1",
    random_seed: int = 42,
    with_checkpoint: bool = True,
    with_benchmark: bool = False,
    metrics_steps: list[tuple[int, float, float, float]] | None = None,
) -> str:
    """Register a COMPLETED/MEASURED experiment with optional checkpoint and metrics."""
    registry.register_experiment(
        experiment_id=experiment_id,
        purpose="test measured run",
        hypothesis=hypothesis,
        config_hash=CONFIG_HASH,
        git_sha=GIT_SHA,
        git_branch="main",
        random_seed=random_seed,
        tags=["iveri", "test"],
        provenance_label="MEASURED",
        status="COMPLETED",
    )

    if with_checkpoint:
        registry.register_checkpoint(
            checkpoint_id=f"ckpt_{experiment_id}",
            experiment_id=experiment_id,
            step=100,
            path=f"checkpoints/{experiment_id}/final.pt",
            chk_hash=CHECKPOINT_HASH,
            parameters_count=10_480_256,
        )
        registry.log_release_manifest(
            release_id=f"rel_{experiment_id}",
            experiment_id=experiment_id,
            release_hash=RELEASE_HASH,
            metadata={"provenance_label": "MEASURED"},
            env_info={"git_sha": GIT_SHA, "git_branch": "main"},
        )

    step_rows = metrics_steps or [(100, 1.5, 1.4, 4.0)]
    for step, train_loss, val_loss, perplexity in step_rows:
        registry.log_metrics(
            experiment_id=experiment_id,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            accuracy=0.8,
            provenance_label="MEASURED",
        )

    if with_benchmark:
        registry.register_benchmark(
            benchmark_id="HumanEval",
            name="HumanEval",
            version="v1.0",
            source="OpenAI",
            dataset_revision="main",
            prompt_suite_version="3A-v1.0",
            hash_sha256=BENCHMARK_REGISTRY_HASH,
            num_prompts=17,
            evaluation_parameters={},
        )
        run_id = f"bench_{experiment_id}"
        registry.log_benchmark_run(
            run_id=run_id,
            experiment_id=experiment_id,
            benchmark_id="HumanEval",
            step=100,
            score=0.85,
            provenance_label="MEASURED",
        )
        registry.log_benchmark_integrity(
            run_id=run_id,
            prompt_hash_ok=True,
            template_hash_ok=True,
            system_prompt_hash_ok=True,
            fewshot_hash_ok=True,
            generation_params_hash_ok=True,
            dataset_hash_ok=True,
            reproducibility_ok=True,
            integrity_report_path="reports/integrity.md",
        )

    return experiment_id


def seed_provenance_chain_experiments(registry: ExperimentRegistry) -> list[str]:
    """Seed ten hypothesis-linked COMPLETED/MEASURED experiments for replay verification."""
    exp_ids: list[str] = []
    registry.register_benchmark(
        benchmark_id="HumanEval",
        name="HumanEval",
        version="v1.0",
        source="OpenAI",
        dataset_revision="main",
        prompt_suite_version="3A-v1.0",
        hash_sha256=BENCHMARK_REGISTRY_HASH,
        num_prompts=17,
        evaluation_parameters={},
    )
    for i in range(1, 11):
        hyp = f"H{i}"
        exp_id = f"IVERI_Phase5_pretrain_Seed42_IVERI_Run{i:03d}"
        seed_measured_experiment(
            registry,
            exp_id,
            hypothesis=hyp,
            with_checkpoint=True,
            with_benchmark=True,
        )
        exp_ids.append(exp_id)
    return exp_ids
