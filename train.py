# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

"""
IVERI CORE — Main training entry point.

Usage:
    python train.py --config configs/base_config.py --verification-level 1
    python train.py --verification-level 2
    python train.py --run-baseline --verification-level 2

This is the primary entry point for all training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import IVERI_VERSION, PROJECT_NAME
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} v{IVERI_VERSION} — Training Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Uses base config if not specified.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--verification-level",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Three-tier verification: 1 (20 steps), 2 (100 steps), 3 (1000 steps).",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Train a baseline vanilla Transformer of similar size for comparison.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        help="Target pretraining dataset (e.g. tinystories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and model but do not train. Useful for validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    logger.info(f"{PROJECT_NAME} v{IVERI_VERSION}")
    logger.info("=" * 60)

    # Config loading
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        logger.info(f"Loading config from: {config_path}")
        from configs.base_config import IVERIConfig

        config = IVERIConfig.load(config_path)
    else:
        logger.info("Using default base config (10M nano prototype)")
        from configs.base_config import get_base_config

        config = get_base_config()

    # Device override
    if args.device is not None:
        config.hardware.device = args.device
        logger.info(f"Device override: {args.device}")

    # Scale down workload on CPU for verification levels to make them fast and practical
    if "cpu" in config.hardware.device.lower():
        logger.info("CPU execution detected. Adjusting batch size to 4, sequence length to 128, gradient accumulation to 1, disabling multiprocessing (num_workers=0), and scaling model size to (32d, 2L) for efficient CPU pretraining verification.")
        config.training.batch_size = min(config.training.batch_size, 4)
        config.training.seq_len = min(config.training.seq_len, 128)
        config.training.gradient_accumulation = 1
        config.hardware.num_workers = 0
        
        # Model dimension scaling overrides
        config.model.hidden_dim = 32
        config.model.num_layers = 2
        config.model.num_heads = 2
        config.model.num_experts = 2
        config.model.num_active_experts = 1
        config.model.titans_memory_dim = 16

    logger.info(f"Config loaded: {config.model.hidden_dim}d, {config.model.num_layers}L, batch_size={config.training.batch_size}, seq_len={config.training.seq_len}")

    if args.dry_run:
        logger.info("Dry run complete. Config is valid.")
        return

    # Run pretraining
    from training.pretrain_runner import run_pretraining

    logger.info(
        f"Starting Pretraining Run. Verification Level: {args.verification_level}, "
        f"Baseline Mode: {args.run_baseline}, Dataset: {args.dataset}"
    )
    
    results = run_pretraining(
        config=config,
        verification_level=args.verification_level,
        run_baseline=args.run_baseline,
        dataset_name=args.dataset,
    )
    
    logger.info("=" * 60)
    logger.info("Training Run completed successfully.")
    logger.info(f"Final Loss: {results['final_loss']:.4f}")
    logger.info(f"Final Val Loss: {results['final_val_loss']:.4f}")
    logger.info(f"Final Perplexity: {results['final_perplexity']:.2f}")
    logger.info(f"Checkpoints saved to: {results['checkpoint_dir']}")


if __name__ == "__main__":
    main()
