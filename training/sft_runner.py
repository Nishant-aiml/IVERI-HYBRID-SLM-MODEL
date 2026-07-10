# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""SFT training runner for IVERI CORE Phase 3.2 instruction tuning.

Orchestrates the full supervised fine-tuning pipeline:
  1. Load pretrained checkpoint from Phase 3.1
  2. Build SFT dataset from data registry
  3. Run fine-tuning loop with masked cross-entropy loss
  4. Evaluate on prompt suite
  5. Save checkpoints and reports

Mirrors ``training/pretrain_runner.py`` in structure but targets Stage 2 SFT.

Examples
--------
>>> from configs.base_config import IVERIConfig
>>> from training.sft_runner import run_sft
>>> config = IVERIConfig()
>>> results = run_sft(config, verification_level=2)  # doctest: +SKIP
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
from model.iveri_core import IVERIModel
from training.checkpointing import save_checkpoint, load_checkpoint
from training.convergence import ConvergenceAnalyzer
from training.experiment_manager import ExperimentManager
from training.instruction_dataset import InstructionDatasetLoader
from training.loss_mask import apply_mask_to_loss
from training.loss_monitor import LossMonitor
from training.model_selection import SFTCheckpointSelector
from training.sft_dataset import SFTByteDataset, make_sft_dataloader
from training.trainer import Trainer

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_val(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _assert_finite(model: Any, loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise ValueError(f"SFT instability: loss is not finite: {loss.item()}")
    for name, p in model.named_parameters():
        if p.requires_grad and not torch.isfinite(p.data).all():
            raise ValueError(f"SFT instability: weights '{name}' contain NaN/Inf.")


# ── Verification step mapping ──────────────────────────────────────────────

_VERIFICATION_STEPS: dict[int, int] = {1: 20, 2: 100, 3: 1000}


# ── Main SFT runner ────────────────────────────────────────────────────────


def run_sft(
    config: IVERIConfig,
    verification_level: int = 2,
    dataset_name: str = "magpie_pro",
    pretrained_checkpoint: str | None = None,
    train_ds_override: Any | None = None,
    val_ds_override: Any | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute SFT instruction tuning on top of a pretrained IVERI checkpoint.

    Parameters
    ----------
    config:
        Master IVERI configuration.
    verification_level:
        1 = 20 steps, 2 = 100 steps, 3 = 1000 steps.
    dataset_name:
        SFT dataset name (registry key). Used unless ``train_ds_override`` is set.
    pretrained_checkpoint:
        Path to Phase 3.1 checkpoint.  Overrides ``config.instruction.pretrained_checkpoint``.
    train_ds_override:
        Direct dataset override for testing (skips registry/validation).
    val_ds_override:
        Direct val dataset override.
    seed:
        Random seed.

    Returns
    -------
    dict[str, Any]
        Keys: ``final_loss``, ``final_val_loss``, ``final_perplexity``,
        ``checkpoint_dir``, ``analysis``, ``generation_results``.
    """
    from evaluation.evaluator import Evaluator
    from evaluation.sft_evaluator import SFTEvaluator
    from evaluation.prompt_suite import PromptSuite
    from evaluation.response_inspector import ResponseInspector

    # ── 1. Step budget ─────────────────────────────────────────────────
    max_steps = _VERIFICATION_STEPS.get(verification_level, 100)
    _set_val(config.training, "max_steps", max_steps)

    eval_every = min(_get_val(config.logging, "eval_every", 500), max(max_steps // 2, 1))
    save_every = min(_get_val(config.logging, "save_every", 1000), max(max_steps // 2, 1))
    _set_val(config.logging, "eval_every", eval_every)
    _set_val(config.logging, "save_every", save_every)

    instr_cfg = getattr(config, "instruction", None)

    # Resolve train_on_prompt
    train_on_prompt = _get_val(instr_cfg, "train_on_prompt", False) if instr_cfg else False

    # Resolve conversation template
    conv_template = _get_val(instr_cfg, "conversation_template", "alpaca") if instr_cfg else "alpaca"

    # Resolve pretrained checkpoint
    ckpt_path = pretrained_checkpoint or (
        _get_val(instr_cfg, "pretrained_checkpoint", "") if instr_cfg else ""
    )

    # ── 2. Experiment setup ────────────────────────────────────────────
    exp_manager = ExperimentManager(config, run_name=f"iveri_stage2_sft_lvl{verification_level}")
    exp_manager.set_seed(seed)

    live_dir = Path("reports/live_training")
    live_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Dataset assembly ────────────────────────────────────────────
    if train_ds_override is not None:
        train_ds = train_ds_override
        val_ds = val_ds_override
        dataset_version = "mock-version"
        dataset_hash = "mock-hash"
    else:
        loader = InstructionDatasetLoader(config)
        train_ds = loader.load(
            dataset_name,
            split="train",
            seq_len=config.training.seq_len,
            train_on_prompt=train_on_prompt,
            max_samples=max_steps * config.training.batch_size * 4,  # Cap for verification
        )
        val_ds = loader.load(
            dataset_name,
            split="val",
            seq_len=config.training.seq_len,
            train_on_prompt=train_on_prompt,
            max_samples=max_steps * config.training.batch_size,
        )
        dataset_version = dataset_name + "-v1"
        dataset_hash = "pipeline-hash"

    exp_manager.setup_run(seed=seed, dataset_version=dataset_version, dataset_hash=dataset_hash)
    exp_manager.log_sft_metadata(
        dataset_mixture=[dataset_name],
        conversation_template=conv_template,
        train_on_prompt=train_on_prompt,
        pretrained_checkpoint=ckpt_path,
    )

    # ── 4. DataLoaders ─────────────────────────────────────────────────
    num_workers = _get_val(config.hardware, "num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    ) if val_ds and len(val_ds) > 0 else None

    # ── 5. Model ───────────────────────────────────────────────────────
    logger.info("Instantiating IVERI model for SFT (Level %d)...", verification_level)
    model = IVERIModel(config)
    device = torch.device(_get_val(config.hardware, "device", "cpu"))
    model.to(device)

    # Load pretrained checkpoint if provided
    if ckpt_path and Path(ckpt_path).exists():
        logger.info("Loading pretrained checkpoint: %s", ckpt_path)
        try:
            load_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
            )
            logger.info("Pretrained checkpoint loaded successfully.")
        except Exception as exc:
            logger.warning("Could not load checkpoint '%s': %s. Starting from scratch.", ckpt_path, exc)
    elif ckpt_path:
        logger.warning("Checkpoint path '%s' does not exist. Starting from scratch.", ckpt_path)
    else:
        logger.info("No pretrained checkpoint specified. Training from random init.")

    # ── 6. Loss Monitor & Convergence ──────────────────────────────────
    loss_monitor = LossMonitor(config, log_dir=exp_manager.experiment_dir)
    loss_monitor.register_activation_hooks(model)
    convergence_analyzer = ConvergenceAnalyzer(window_size=max_steps)

    # ── 7. Evaluator & Inspector ───────────────────────────────────────
    base_evaluator = Evaluator(model, config, val_dataloader=val_loader)
    sft_evaluator = SFTEvaluator(base_evaluator, config, ResponseInspector())
    prompt_suite = PromptSuite()

    # ── 8. Trainer ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    # ── 9. Checkpoint Selector ─────────────────────────────────────────
    selector = SFTCheckpointSelector(log_dir=exp_manager.experiment_dir)

    # ── 10. Training loop ──────────────────────────────────────────────
    model.train()
    step = 0
    epoch = 0

    _init_live_csv(live_dir, prefix="sft")

    epoch_iterator = iter(train_loader)
    initial_val_metrics: dict[str, float] = {}

    # Initial evaluation
    if val_loader:
        initial_val_metrics = sft_evaluator.evaluate_sft(val_loader)
        logger.info(
            "[SFT Initial Eval] val_loss=%.4f perplexity=%.2f top1=%.2f%%",
            initial_val_metrics.get("sft/val_loss", 0.0),
            initial_val_metrics.get("sft/perplexity", 0.0),
            initial_val_metrics.get("sft/top1_accuracy", 0.0) * 100,
        )
        model.train()

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

        # Unpack SFT batch: (x, y, loss_mask)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, loss_mask = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            loss_mask = torch.ones_like(y, dtype=torch.bool)
        elif isinstance(batch, dict):
            x = batch["input_ids"]
            y = batch["labels"]
            loss_mask = batch.get("loss_mask", torch.ones_like(y, dtype=torch.bool))
        else:
            x, y = batch, batch
            loss_mask = torch.ones_like(y, dtype=torch.bool)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        # Forward pass
        trainer.optimizer.zero_grad(set_to_none=True)
        with trainer.precision_handler.autocast_context():
            outputs = model(x, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            aux_loss = (
                outputs.get("aux_loss", torch.tensor(0.0, device=device))
                if isinstance(outputs, dict)
                else torch.tensor(0.0, device=device)
            )

            # Masked SFT loss
            B, S, V = logits.shape
            flat_logits = logits.reshape(-1, V)
            flat_y = y.reshape(-1)
            flat_mask = loss_mask.reshape(-1)

            if flat_mask.any():
                # Compute CE loss only on unmasked (response) positions
                per_token_loss = torch.nn.functional.cross_entropy(
                    flat_logits, flat_y, reduction="none"
                )
                n_active = flat_mask.float().sum().clamp(min=1.0)
                loss = (per_token_loss * flat_mask.float()).sum() / n_active
            else:
                # Fallback: full sequence loss (all tokens masked — shouldn't happen)
                loss = torch.nn.functional.cross_entropy(flat_logits, flat_y)

            composite_loss = loss + 0.01 * aux_loss
            scaled_loss = trainer.precision_handler.scale_loss(composite_loss)

        # Backward
        scaled_loss.backward()

        # Optimizer step
        trainer.precision_handler.step_optimizer(
            optimizer=trainer.optimizer,
            max_norm=config.training.grad_clip,
        )

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        elapsed_step = time.perf_counter() - t_step0
        step += 1

        # Track training instability and divergence diagnostics
        trainer.instability_tracker.step(step)


        # Numerical health
        _assert_finite(model, loss)

        # Monitors
        loss_monitor.update_loss(loss.item(), is_val=False)
        num_tokens = x.numel()
        num_patches = 0
        if isinstance(outputs, dict) and "boundary_map" in outputs:
            num_patches = outputs["boundary_map"].sum().item()

        convergence_analyzer.update(loss.item(), elapsed_step, num_tokens, num_patches)
        grad_health = loss_monitor.track_gradient_health(model, step)

        perf = convergence_analyzer.compute_throughput(
            model,
            batch_size=x.size(0),
            seq_len=x.size(1),
            last_step_time=elapsed_step,
        )

        lr = (
            trainer.scheduler.get_last_lr()[0]
            if trainer.scheduler is not None
            else config.training.learning_rate
        )

        trainer.logger.log(
            {
                "sft/train_loss": loss.item(),
                "sft/train_perplexity": math.exp(min(loss.item(), 50)),
                "sft/learning_rate": lr,
                "sft/bpb": loss.item() / math.log(2),
                "sft/masked_tokens": int(flat_mask.sum().item()),
                **perf,
                **grad_health,
            },
            step=step,
        )

        _append_live_csv(live_dir, step, loss.item(), initial_val_metrics, lr, perf)

        # Periodic evaluation
        if step % eval_every == 0 and val_loader:
            val_metrics = sft_evaluator.evaluate_sft(val_loader)
            loss_monitor.update_loss(val_metrics["sft/val_loss"], is_val=True)
            logger.info(
                "[SFT Step %d] train_loss=%.4f val_loss=%.4f perplexity=%.2f "
                "top1=%.2f%% bpb=%.3f",
                step,
                loss.item(),
                val_metrics["sft/val_loss"],
                val_metrics["sft/perplexity"],
                val_metrics["sft/top1_accuracy"] * 100,
                val_metrics["sft/bits_per_byte"],
            )
            initial_val_metrics = val_metrics
            model.train()

        # Periodic checkpoint
        if step % save_every == 0:
            chk_path = exp_manager.experiment_dir / f"sft_checkpoint_{step}.pt"
            checkpoint_meta = {
                "training_stage": "Stage 2 SFT",
                "dataset": dataset_name,
                "conversation_template": conv_template,
                "train_on_prompt": train_on_prompt,
                "pretrained_checkpoint": ckpt_path,
                "architecture_version": model.architecture_version,
                "seed": seed,
            }
            save_checkpoint(
                path=chk_path,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.precision_handler.scaler,
                step=step,
                epoch=epoch,
                metrics={
                    "loss": loss.item(),
                    "perplexity": math.exp(min(loss.item(), 50)),
                    **checkpoint_meta,
                },
                config=config,
            )
            val_l = initial_val_metrics.get("sft/val_loss", loss.item())
            val_p = initial_val_metrics.get("sft/perplexity", math.exp(min(loss.item(), 50)))
            selector.register_checkpoint(
                path=chk_path,
                step=step,
                train_loss=loss.item(),
                val_loss=val_l,
                perplexity=val_p,
                metadata=checkpoint_meta,
            )
            exp_manager.save_resume_metadata(step, epoch, chk_path)
            _update_checkpoint_history_md(live_dir, selector)

    # ── 11. Cleanup ────────────────────────────────────────────────────
    loss_monitor.remove_hooks()
    trainer.instability_tracker.remove_hooks()
    trainer.shutdown_logger()


    # ── 12. Final generation evaluation ───────────────────────────────
    model.eval()
    logger.info("Running final prompt suite evaluation...")
    # Limit generation length for level-1 (20-step test runs) to avoid CPU timeout —
    # full-length generation is only meaningful after real SFT training.
    _max_gen_bytes = 2 if verification_level == 1 else _get_val(
        getattr(config, "instruction", None), "max_new_bytes", 128
    )
    gen_results = sft_evaluator.evaluate_prompt_suite(
        prompt_suite,
        max_new_bytes=_max_gen_bytes,
        temperature=config.evaluation.generation_temperature,
        top_k=config.evaluation.generation_top_k,
        seed=seed,
    )


    final_analysis = convergence_analyzer.analyze()
    elapsed_total = time.perf_counter() - t_start
    logger.info("SFT Level %d completed in %.2fs.", verification_level, elapsed_total)

    return {
        "final_loss": loss.item(),
        "final_val_loss": initial_val_metrics.get("sft/val_loss", loss.item()),
        "final_perplexity": initial_val_metrics.get(
            "sft/perplexity", math.exp(min(loss.item(), 50))
        ),
        "checkpoint_dir": str(exp_manager.experiment_dir.as_posix()),
        "analysis": final_analysis,
        "generation_results": gen_results,
    }


# ── CSV utilities ─────────────────────────────────────────────────────────


def _init_live_csv(live_dir: Path, prefix: str = "sft") -> None:
    with open(live_dir / f"{prefix}_loss.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss"])
    with open(live_dir / f"{prefix}_perplexity.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "train_perplexity", "val_perplexity"])
    with open(live_dir / f"{prefix}_lr.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "learning_rate"])


def _append_live_csv(
    live_dir: Path,
    step: int,
    train_loss: float,
    val_metrics: dict[str, float],
    lr: float,
    perf: dict[str, float],
) -> None:
    val_loss = val_metrics.get("sft/val_loss", float("nan"))
    train_perp = math.exp(min(train_loss, 50))
    val_perp = val_metrics.get("sft/perplexity", float("nan"))

    with open(live_dir / "sft_loss.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}"])
    with open(live_dir / "sft_perplexity.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{train_perp:.6f}", f"{val_perp:.6f}"])
    with open(live_dir / "sft_lr.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step, f"{lr:.8f}"])


def _update_checkpoint_history_md(live_dir: Path, selector: Any) -> None:
    lines = [
        "# IVERI CORE — SFT Checkpoint History",
        "",
        "| Step | Train Loss | Val Loss | Perplexity | Path |",
        "|---|---|---|---|---|",
    ]
    for c in selector.get_history():
        lines.append(
            f"| {c['step']} | {c['train_loss']:.4f} | {c['val_loss']:.4f} | "
            f"{c['perplexity']:.2f} | `{c['path']}` |"
        )
    with open(live_dir / "sft_checkpoint_history.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
