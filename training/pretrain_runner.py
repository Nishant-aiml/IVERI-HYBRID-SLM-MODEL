# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Pretraining runner orchestrating pretraining iterations and convergence validation.

Supports three-tier verification (20/100/1000 steps), baseline transformer runs,
numerical health checks, and live learning curve CSV snapshots.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from evaluation.evaluator import Evaluator
from evaluation.pretraining_eval import PretrainingEvaluator
from evaluation.generation_inspector import GenerationInspector
from baselines.baseline_transformer import BaselineTransformer
from model.iveri_core import IVERIModel
from training.checkpointing import save_checkpoint, load_checkpoint
from training.convergence import ConvergenceAnalyzer
from training.curriculum import CurriculumScheduler
from training.experiment_manager import ExperimentManager
from training.loss_monitor import LossMonitor
from training.pretraining_dataset import PretrainingDatasetLoader
from training.trainer import Trainer

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_val(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def run_pretraining(
    config: IVERIConfig,
    verification_level: int = 2,
    run_baseline: bool = False,
    dataset_name: str = "tinystories",
    train_ds_override: Any | None = None,
    val_ds_override: Any | None = None,
) -> dict[str, Any]:
    """Execute standard foundation pretraining.

    Args:
        config: Configuration parameters.
        verification_level: 1 (20 steps), 2 (100 steps), 3 (1000 steps).
        run_baseline: Whether to train baseline vanilla transformer.
        dataset_name: Target pretraining dataset (e.g. tinystories).
        train_ds_override: Optional override for test training dataset.
        val_ds_override: Optional override for test validation dataset.
    """
    # 1. Resolve steps from verification level
    max_steps = 100
    if verification_level == 1:
        max_steps = 20
    elif verification_level == 2:
        max_steps = 100
    elif verification_level == 3:
        max_steps = 1000

    # Ensure config matches verification steps
    _set_val(config.training, "max_steps", max_steps)
    
    # Set evaluation and checkpoint cadence safely
    eval_every = min(_get_val(config.logging, "eval_every", 500), max_steps // 2)
    save_every = min(_get_val(config.logging, "save_every", 1000), max_steps // 2)
    _set_val(config.logging, "eval_every", eval_every)
    _set_val(config.logging, "save_every", save_every)

    # Resolve seed
    seed = _get_val(config.training, "seed", 42)

    # 2. Experiment Setup
    exp_manager = ExperimentManager(config, run_name=f"iveri_stage1_lvl{verification_level}")
    exp_manager.set_seed(seed)

    # Setup directories
    live_dir = Path("reports/live_training")
    live_dir.mkdir(parents=True, exist_ok=True)

    # 3. Assemble Dataset
    if train_ds_override is not None:
        train_ds = train_ds_override
        val_ds = val_ds_override
        dataset_version = "mock-version"
        dataset_hash = "mock-hash"
    else:
        # Strict validation ingestion
        loader = PretrainingDatasetLoader(config)
        train_ds = loader.load(dataset_name, split="train")
        val_ds = loader.load(dataset_name, split="val")
        # Load version info
        data_pipeline = getattr(config, "data_pipeline", {})
        report_cfg = _get_val(data_pipeline, "report", {})
        processed_base = Path(_get_val(report_cfg, "processed_data_dir", "data/processed"))
        processed_dir = processed_base / "stage1" / dataset_name
        if not processed_dir.exists():
            processed_dir = processed_base / dataset_name
        version_info = loader.versioner.load_version(processed_dir)
        dataset_version = version_info.version_id
        dataset_hash = version_info.content_hash

    # Setup run log metadata
    exp_manager.setup_run(
        seed=seed,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
        drop_last=True,
    ) if val_ds else None

    # 4. Build Model
    if run_baseline:
        logger.info(f"Instantiating Baseline Vanilla Transformer (Level {verification_level})...")
        model = BaselineTransformer(config)
    else:
        logger.info(f"Instantiating IVERI Model (Level {verification_level})...")
        model = IVERIModel(config)

    # 5. Initialize Loss Monitor & Convergence
    loss_monitor = LossMonitor(config, log_dir=exp_manager.experiment_dir)
    loss_monitor.register_activation_hooks(model)
    convergence_analyzer = ConvergenceAnalyzer(window_size=max_steps)

    # 6. Initialize Evaluator & Generation Inspector
    evaluator_base = Evaluator(model, config, val_dataloader=val_loader)
    pretrain_evaluator = PretrainingEvaluator(evaluator_base)
    inspector = GenerationInspector(config, log_dir=live_dir)

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    device = torch.device(config.hardware.device)
    model.to(device)

    # Checkpoint indexing
    from training.model_selection import CheckpointSelector
    selector = CheckpointSelector(log_dir=exp_manager.experiment_dir)

    # 8. Training loop
    model.train()
    step = 0
    epoch = 0

    # Initialize live curves CSV files
    _init_live_csv(live_dir)

    epoch_iterator = iter(train_loader)

    # Log initial evaluation
    initial_val_metrics = {}
    if val_loader:
        initial_val_metrics = pretrain_evaluator.evaluate_pretraining(val_loader)
        logger.info(
            f"[Initial Eval] val_loss={initial_val_metrics['val_loss']:.4f} "
            f"perplexity={initial_val_metrics['perplexity']:.2f} BPB={initial_val_metrics['bits_per_byte']:.3f}"
        )

    t_start = time.perf_counter()

    while step < max_steps:
        t_step0 = time.perf_counter()

        # Grab next batch
        try:
            batch = next(epoch_iterator)
        except StopIteration:
            epoch += 1
            epoch_iterator = iter(train_loader)
            batch = next(epoch_iterator)

        # Ingestion unpack
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch["input_ids"]
            targets = batch["labels"]
        else:
            inputs, targets = batch, batch

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Autocast context
        trainer.optimizer.zero_grad(set_to_none=True)
        with trainer.precision_handler.autocast_context():
            outputs = model(inputs, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=device)) if isinstance(outputs, dict) else torch.tensor(0.0, device=device)

            # Autoregressive loss
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
            composite_loss = loss + 0.01 * aux_loss
            scaled_loss = trainer.precision_handler.scale_loss(composite_loss)

        # Backward
        scaled_loss.backward()

        # Optimizer step
        trainer.precision_handler.step_optimizer(
            optimizer=trainer.optimizer,
            max_norm=config.training.grad_clip,
        )

        # Step Scheduler
        if trainer.scheduler is not None:
            trainer.scheduler.step()

        elapsed_step = time.perf_counter() - t_step0
        step += 1

        # Numerical Stability Check: verify weights, gradients, loss, activations are finite
        _assert_finite_tensors(model, loss)

        # Update Loss Monitor & Convergence
        loss_monitor.update_loss(loss.item(), is_val=False)
        # Count actual processed tokens/patches
        num_tokens = inputs.numel()
        # Heuristic patch count for BLT
        num_patches = 0
        if isinstance(outputs, dict) and "boundary_map" in outputs:
            num_patches = outputs["boundary_map"].sum().item()

        convergence_analyzer.update(loss.item(), elapsed_step, num_tokens, num_patches)

        # Track Gradient Health
        grad_health = loss_monitor.track_gradient_health(model, step)

        # Compute Throughput & Performance
        perf = convergence_analyzer.compute_throughput(
            model,
            batch_size=inputs.size(0),
            seq_len=inputs.size(1),
            last_step_time=elapsed_step,
        )

        # Log to local and experiment logger
        lr = (
            trainer.scheduler.get_last_lr()[0]
            if trainer.scheduler is not None
            else config.training.learning_rate
        )
        trainer.logger.log({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(loss.item()) if loss.item() < 50 else float("inf"),
            "train/learning_rate": lr,
            "train/bpb": loss.item() / math.log(2),
            **perf,
            **grad_health,
        }, step=step)

        # Save snapshots to CSV files
        _append_live_curves(
            live_dir=live_dir,
            step=step,
            train_loss=loss.item(),
            val_loss=initial_val_metrics.get("val_loss", float("nan")),
            train_perp=math.exp(loss.item()) if loss.item() < 50 else float("inf"),
            val_perp=initial_val_metrics.get("perplexity", float("nan")),
            lr=lr,
            perf=perf,
        )

        # Periodic Evaluation
        if step % config.logging.eval_every == 0 and val_loader:
            val_metrics = pretrain_evaluator.evaluate_pretraining(val_loader)
            loss_monitor.update_loss(val_metrics["val_loss"], is_val=True)
            logger.info(
                f"[Step {step}] train_loss={loss.item():.4f} val_loss={val_metrics['val_loss']:.4f} "
                f"perplexity={val_metrics['perplexity']:.2f} BPB={val_metrics['bits_per_byte']:.3f} "
                f"accuracy={val_metrics['top1_accuracy']:.2%} top5={val_metrics['top5_accuracy']:.2%}"
            )
            # Update live curves with latest validation
            initial_val_metrics = val_metrics

        # Periodic Generation Inspection
        if step % config.logging.log_every == 0:
            insp_results = inspector.inspect(
                model=model,
                step=step,
                temperature=config.evaluation.generation_temperature,
                seed=seed,
            )

        # Periodic Checkpoint saving
        if step % config.logging.save_every == 0:
            chk_path = exp_manager.experiment_dir / f"checkpoint_{step}.pt"
            # Build robust metadata
            checkpoint_metadata = {
                "dataset_version": dataset_version,
                "pipeline_version": loader.versioner.pipeline_version if 'loader' in locals() else "3.0.0",
                "git_commit": exp_manager.metadata_path.exists() and json.loads(exp_manager.metadata_path.read_text()).get("git_commit", "") or "",
                "architecture_version": model.architecture_version,
                "config_hash": loader.versioner.compute_pipeline_hash(config.to_dict()) if 'loader' in locals() else "",
                "training_stage": "Stage 1 Pretraining",
                "seed": seed,
            }

            # Save checkpoint
            save_checkpoint(
                path=chk_path,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.precision_handler.scaler,
                step=step,
                epoch=epoch,
                metrics={"loss": loss.item(), "perplexity": math.exp(loss.item()) if loss.item() < 50 else float("inf"), **checkpoint_metadata},
                config=config,
            )

            # Register with Selector
            val_l = initial_val_metrics.get("val_loss", loss.item())
            val_p = initial_val_metrics.get("perplexity", math.exp(loss.item()) if loss.item() < 50 else float("inf"))
            selector.register_checkpoint(
                path=chk_path,
                step=step,
                train_loss=loss.item(),
                val_loss=val_l,
                perplexity=val_p,
                metadata=checkpoint_metadata,
            )
            # Save latest resume meta
            exp_manager.save_resume_metadata(step, epoch, chk_path)

            # Update selector history log markdown
            _update_checkpoint_history_markdown(live_dir, selector)

    loss_monitor.remove_hooks()
    trainer.shutdown_logger()

    final_analysis = convergence_analyzer.analyze()
    logger.info(f"Completed pretraining level {verification_level} in {time.perf_counter() - t_start:.2f}s.")

    return {
        "final_loss": loss.item(),
        "final_val_loss": initial_val_metrics.get("val_loss", loss.item()),
        "final_perplexity": initial_val_metrics.get("perplexity", math.exp(loss.item()) if loss.item() < 50 else float("inf")),
        "checkpoint_dir": str(exp_manager.experiment_dir.as_posix()),
        "analysis": final_analysis,
    }


def _assert_finite_tensors(model: nn.Module, loss: torch.Tensor) -> None:
    """Numerical stability assertions. Checks loss, gradients, and weights for NaNs/Infs."""
    if not torch.isfinite(loss):
        raise ValueError(f"Numerical instability: loss is not finite: {loss.item()}")

    for name, p in model.named_parameters():
        if p.requires_grad:
            if not torch.isfinite(p.data).all():
                raise ValueError(f"Numerical instability: weights of parameter '{name}' contain NaN/Inf.")
            if p.grad is not None and not torch.isfinite(p.grad.data).all():
                raise ValueError(f"Numerical instability: gradients of parameter '{name}' contain NaN/Inf.")


def _init_live_csv(live_dir: Path) -> None:
    """Initialize learning curve CSV files."""
    # loss.csv
    with open(live_dir / "loss.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss"])
    # perplexity.csv
    with open(live_dir / "perplexity.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_perplexity", "val_perplexity"])
    # learning_rate.csv
    with open(live_dir / "learning_rate.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "learning_rate"])
    # throughput.csv
    with open(live_dir / "throughput.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "tokens_per_sec", "bytes_per_sec", "patches_per_sec", "samples_per_sec"])


def _append_live_curves(
    live_dir: Path,
    step: int,
    train_loss: float,
    val_loss: float,
    train_perp: float,
    val_perp: float,
    lr: float,
    perf: dict[str, float],
) -> None:
    """Append values to CSV snapshots."""
    # loss
    with open(live_dir / "loss.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}"])
    # perplexity
    with open(live_dir / "perplexity.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{train_perp:.6f}", f"{val_perp:.6f}"])
    # lr
    with open(live_dir / "learning_rate.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{lr:.8f}"])
    # throughput
    with open(live_dir / "throughput.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            step,
            f"{perf.get('performance/tokens_per_sec', 0.0):.2f}",
            f"{perf.get('performance/bytes_per_sec', 0.0):.2f}",
            f"{perf.get('performance/patches_per_sec', 0.0):.2f}",
            f"{perf.get('performance/samples_per_sec', 0.0):.2f}",
        ])


def _update_checkpoint_history_markdown(live_dir: Path, selector: Any) -> None:
    """Write the checkpoint history summary to checkpoint_history.md."""
    lines = [
        "# IVERI CORE — Checkpoint History Summary",
        "",
        "| Step | Train Loss | Val Loss | Perplexity | Path |",
        "|---|---|---|---|---|",
    ]
    for c in selector.get_history():
        lines.append(
            f"| {c['step']} | {c['train_loss']:.4f} | {c['val_loss']:.4f} | "
            f"{c['perplexity']:.2f} | `{c['path']}` |"
        )
    with open(live_dir / "checkpoint_history.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
