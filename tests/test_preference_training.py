# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for Phase 3.4 Preference Optimization.

Verifies configuration, datasets, formatter, losses, reference model manager,
checkpoint ranking, alignment prompt suite, inspector, and runner.
"""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from configs.preference_config import PreferenceConfig
from core.exceptions import ConfigError
from core.constants import ARCHITECTURE_VERSION
from training.preference_dataset import PreferenceDatasetLoader, PreferenceByteDataset
from training.preference_formatter import PreferenceFormatter, FormattedPreferencePair
from training.preference_loss import PreferenceLoss, compute_logps
from training.reference_model import ReferenceModelManager, verify_parameter_equality, verify_checkpoint_compatibility
from training.model_selection import PreferenceCheckpointSelector
from evaluation.alignment_prompt_suite import AlignmentPromptSuite
from evaluation.alignment_inspector import AlignmentInspector
from evaluation.preference_benchmark import PreferenceBenchmarkRunner
from training.preference_runner import run_preference_training

# Helper mock model
class MockModel(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        # Dummy parameter to make it look like a model
        self.weight = nn.Parameter(torch.ones(1, hidden_dim))

    def forward(self, input_ids, return_dict=True):
        embeds = self.embedding(input_ids)
        logits = self.lm_head(embeds)
        if return_dict:
            return {"logits": logits, "loss": logits.mean()}
        return logits


# ── Test 1: Config Defaults ──────────────────────────────────────────
def test_preference_config_defaults():
    cfg = PreferenceConfig()
    assert not cfg.enabled
    assert cfg.algorithm == "dpo"
    assert cfg.beta == 0.1
    assert cfg.reference_device == "cpu"
    assert cfg.label_smoothing == 0.1
    assert cfg.ipo_gamma == 2.0


# ── Test 2: Config Validation ────────────────────────────────────────
def test_preference_config_validation():
    with pytest.raises(ConfigError):
        PreferenceConfig(algorithm="invalid_alg")
    with pytest.raises(ConfigError):
        PreferenceConfig(reference_device="invalid_dev")
    with pytest.raises(ConfigError):
        PreferenceConfig(beta=-0.5)
    with pytest.raises(ConfigError):
        PreferenceConfig(label_smoothing=0.6)
    with pytest.raises(ConfigError):
        PreferenceConfig(max_sequence_length=0)


# ── Test 3: Config From Dict ──────────────────────────────────────────
def test_preference_config_from_dict():
    cfg_dict = {
        "enabled": True,
        "algorithm": "simpo",
        "beta": 0.2,
        "label_smoothing": 0.2,
        "ipo_gamma": 2.5
    }
    # Test safe instantiation
    cfg = PreferenceConfig(**cfg_dict)
    assert cfg.enabled
    assert cfg.algorithm == "simpo"
    assert cfg.beta == 0.2
    assert cfg.label_smoothing == 0.2
    assert cfg.ipo_gamma == 2.5


# ── Test 4: Dataset Ingestion License Validation ──────────────────────
def test_preference_dataset_loader_license():
    config = IVERIConfig()
    loader = PreferenceDatasetLoader(config)
    
    # Test valid licenses
    valid_entry = type("Entry", (), {"name": "test", "license_id": "Apache-2.0"})()
    loader._check_license(valid_entry)  # should not raise
    
    # Test invalid license
    invalid_entry = type("Entry", (), {"name": "test", "license_id": "L-GPL"})()
    with pytest.raises(RuntimeError):
        loader._check_license(invalid_entry)


# ── Test 5: Dataset Ingestion Stage Verification ──────────────────────
def test_preference_dataset_loader_stage():
    config = IVERIConfig()
    loader = PreferenceDatasetLoader(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        version_file = tmp_path / "VERSION.json"
        
        # Valid stage 4
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump({"stage": 4}, f)
        loader._validate_stage(tmp_path, "test_dataset") # should not raise

        # Invalid stage 3
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump({"stage": 3}, f)
        with pytest.raises(RuntimeError):
            loader._validate_stage(tmp_path, "test_dataset")


# ── Test 6: Preference Formatter Alpaca-style ─────────────────────────
def test_preference_formatter_alpaca():
    formatter = PreferenceFormatter()
    sample = {
        "instruction": "Test instruction",
        "input": "Test input",
        "chosen": "Test chosen response",
        "rejected": "Test rejected response"
    }
    pair = formatter.format_pair(sample)
    
    assert b"Test instruction" in pair.prompt_bytes
    assert b"Test input" in pair.prompt_bytes
    assert b"### Response:\n" in pair.prompt_bytes
    assert pair.chosen_bytes.startswith(b"Test chosen response")
    assert pair.rejected_bytes.startswith(b"Test rejected response")


# ── Test 7: Preference Formatter Chat-style (Messages list) ───────────
def test_preference_formatter_chat():
    formatter = PreferenceFormatter()
    sample = {
        "chosen": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."}
        ],
        "rejected": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 5."}
        ]
    }
    pair = formatter.format_pair(sample)
    
    assert b"What is 2+2?" in pair.prompt_bytes
    assert b"### Response:\n" in pair.prompt_bytes
    assert pair.chosen_bytes.startswith(b"It is 4.")
    assert pair.rejected_bytes.startswith(b"It is 5.")


# ── Test 8: compute_logps Mask Alignment ─────────────────────────────
def test_compute_logps():
    # shape (B=2, S=3, V=4)
    logits = torch.tensor([
        [[0.1, 0.5, 0.2, 0.2], [0.9, 0.05, 0.05, 0.0], [0.1, 0.1, 0.1, 0.7]],
        [[0.2, 0.2, 0.4, 0.2], [0.1, 0.1, 0.8, 0.0], [0.3, 0.3, 0.1, 0.3]]
    ])
    labels = torch.tensor([
        [1, 0, 3],
        [2, 2, 0]
    ])
    mask = torch.tensor([
        [True, False, True],
        [True, True, False]
    ], dtype=torch.bool)

    logps = compute_logps(logits, labels, mask)
    assert logps.shape == (2,)
    assert not torch.isnan(logps).any()


# ── Test 9: compute_logps Length Normalization (SimPO) ────────────────
def test_compute_logps_average():
    logits = torch.tensor([
        [[0.1, 0.5, 0.2, 0.2], [0.9, 0.05, 0.05, 0.0], [0.1, 0.1, 0.1, 0.7]],
    ])
    labels = torch.tensor([[1, 0, 3]])
    mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    logps_sum = compute_logps(logits, labels, mask, average_log_prob=False)
    logps_avg = compute_logps(logits, labels, mask, average_log_prob=True)
    
    # average should be sum divided by 2 response elements
    assert torch.allclose(logps_avg, logps_sum / 2.0)


# ── Test 10: DPO Loss Correctness ─────────────────────────────────────
def test_dpo_loss_correctness():
    loss_fn = PreferenceLoss(algorithm="dpo", beta=0.1)
    
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.5, -2.5])
    ref_rejected = torch.tensor([-2.8, -3.8])

    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    assert loss.dim() == 0  # scalar
    assert chosen_rewards.shape == (2,)
    assert rejected_rewards.shape == (2,)
    assert (chosen_rewards - rejected_rewards).mean().item() > 0.0


# ── Test 11: Conservative DPO Loss (Label Smoothing) ─────────────────
def test_conservative_dpo_loss():
    loss_fn = PreferenceLoss(algorithm="conservative_dpo", beta=0.1, label_smoothing=0.1)
    
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.5, -2.5])
    ref_rejected = torch.tensor([-2.8, -3.8])

    loss, _, _ = loss_fn(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    assert loss.dim() == 0
    assert loss.item() > 0.0


# ── Test 12: IPO Loss (Quadratic regularization) ─────────────────────
def test_ipo_loss():
    loss_fn = PreferenceLoss(algorithm="ipo", beta=0.1)
    
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    ref_chosen = torch.tensor([-1.5, -2.5])
    ref_rejected = torch.tensor([-2.8, -3.8])

    loss, _, _ = loss_fn(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    assert loss.dim() == 0
    assert loss.item() > 0.0


# ── Test 13: SimPO Loss (Length normalized, margin adjusted) ─────────
def test_simpo_loss():
    loss_fn = PreferenceLoss(algorithm="simpo", beta=0.1, ipo_gamma=2.0)
    
    policy_chosen = torch.tensor([-0.5, -1.0])
    policy_rejected = torch.tensor([-1.5, -2.0])

    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen, policy_rejected, None, None
    )
    assert loss.dim() == 0
    assert chosen_rewards.shape == (2,)
    assert rejected_rewards.shape == (2,)


# ── Test 14: Reference Model Manager Loading stub ──────────────────────
def test_reference_model_manager_load():
    config = IVERIConfig()
    manager = ReferenceModelManager(config, torch.device("cpu"))
    manager.load("")
    assert manager.reference_model is None
    assert manager.checkpoint_sha256 == "unknown"


# ── Test 15: Reference Model Manager Parameter Equality ──────────────
def test_reference_model_manager_equality():
    model_a = MockModel()
    model_b = MockModel()
    
    # Load state dict to guarantee complete identity across all parameters
    model_b.load_state_dict(model_a.state_dict())
    
    assert verify_parameter_equality(model_a, model_b)


# ── Test 16: Reference Model Manager Parameter Inequality ────────────
def test_reference_model_manager_inequality():
    model_a = MockModel()
    model_b = MockModel()
    
    model_b.load_state_dict(model_a.state_dict())
    # Mutate one parameter to force difference
    model_b.weight.data.copy_(torch.ones(1, 64) * 9.0)
    
    assert not verify_parameter_equality(model_a, model_b)


# ── Test 17: Checkpoint Compatibility matches expected ───────────────
def test_checkpoint_compatibility_ok():
    config = IVERIConfig()
    config.model.hidden_dim = 256
    config.model.num_layers = 6

    ckpt = {
        "architecture_version": ARCHITECTURE_VERSION,
        "iveri_version": "1.0.0",
        "config": {
            "model": {
                "hidden_dim": 256,
                "num_layers": 6
            }
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        
    try:
        verify_checkpoint_compatibility(f.name, config)
    finally:
        Path(f.name).unlink()


# ── Test 18: Checkpoint Compatibility fails on mismatch ─────────────
def test_checkpoint_compatibility_fail():
    config = IVERIConfig()
    config.model.hidden_dim = 512
    config.model.num_layers = 6

    ckpt = {
        "architecture_version": ARCHITECTURE_VERSION,
        "iveri_version": "1.0.0",
        "config": {
            "model": {
                "hidden_dim": 256,
                "num_layers": 6
            }
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(ckpt, f.name)
        
    try:
        with pytest.raises(ValueError):
            verify_checkpoint_compatibility(f.name, config)
    finally:
        Path(f.name).unlink()


# ── Test 19: Checkpoint Selector Registration ────────────────────────
def test_preference_checkpoint_selector_register():
    with tempfile.TemporaryDirectory() as tmp_dir:
        selector = PreferenceCheckpointSelector(log_dir=Path(tmp_dir))
        selector.register_checkpoint(
            path="dummy_path.pt",
            step=10,
            train_loss=1.5,
            val_loss=1.6,
            perplexity=5.0,
            preference_accuracy=0.85,
            reward_margin=1.2,
            instruction_retention_ok=True,
            coding_retention_ok=True
        )
        assert len(selector.checkpoints) == 1
        assert selector.checkpoints[0]["preference_accuracy"] == 0.85
        assert selector.checkpoints[0]["reward_margin"] == 1.2


# ── Test 20: Checkpoint Selector Ranking priorities ──────────────────
def test_preference_checkpoint_selector_rank():
    with tempfile.TemporaryDirectory() as tmp_dir:
        selector = PreferenceCheckpointSelector(log_dir=Path(tmp_dir))
        
        selector.register_checkpoint(
            path="path_1.pt", step=10, train_loss=1.0, val_loss=1.5, perplexity=4.0,
            preference_accuracy=0.60, reward_margin=0.5, instruction_retention_ok=True, coding_retention_ok=True
        )
        
        selector.register_checkpoint(
            path="path_2.pt", step=20, train_loss=0.8, val_loss=1.3, perplexity=3.5,
            preference_accuracy=0.90, reward_margin=2.0, instruction_retention_ok=True, coding_retention_ok=True
        )

        selector.register_checkpoint(
            path="path_3.pt", step=30, train_loss=0.5, val_loss=1.0, perplexity=2.5,
            preference_accuracy=0.95, reward_margin=2.5, instruction_retention_ok=False, coding_retention_ok=True
        )

        best = selector.get_best_preference_checkpoint()
        assert best is not None
        assert Path(best["path"]).name == "path_2.pt"


# ── Test 21: Alignment Prompt Suite Count ────────────────────────────
def test_alignment_prompt_suite_count():
    suite = AlignmentPromptSuite()
    prompts = suite.get_all()
    assert len(prompts) == 50
    assert suite.get_suite_hash() != ""
    assert len(suite.get_by_category("coding")) >= 5


# ── Test 22: Alignment Inspector Verbosity Warning ────────────────────
def test_alignment_inspector_verbosity():
    inspector = AlignmentInspector(length_threshold=50)
    long_response = " " * 51
    res = inspector.inspect_generations(
        prompts=["Prompt"],
        responses=[long_response]
    )
    assert res.is_anomaly
    assert any("verbosity" in w for w in res.warnings)


# ── Test 23: Alignment Inspector Over-refusal Check ──────────────────
def test_alignment_inspector_refusal():
    inspector = AlignmentInspector()
    res = inspector.inspect_generations(
        prompts=["Do X"],
        responses=["As an AI language model, I cannot help with that ethical request."]
    )
    assert res.refusal_count == 1


# ── Test 24: Preference Benchmark Run ─────────────────────────────────
def test_preference_benchmark_run():
    model = MockModel()
    ref_model = MockModel()
    
    bench = PreferenceBenchmarkRunner(model, ref_model, beta=0.1)
    
    batch = (
        torch.randint(0, 256, (2, 8)),
        torch.randint(0, 256, (2, 8)),
        torch.ones((2, 8), dtype=torch.bool),
        torch.randint(0, 256, (2, 8)),
        torch.randint(0, 256, (2, 8)),
        torch.ones((2, 8), dtype=torch.bool),
    )
    
    from unittest.mock import MagicMock
    precision_handler = MagicMock()
    
    metrics = bench.run_evaluation([batch], torch.device("cpu"), precision_handler)
    assert "benchmark/win_rate" in metrics
    assert "histograms/margin" in metrics


# ── Test 25: E2E Preference Runner Mock Simulation ────────────────────
def test_run_preference_training_mock():
    config = IVERIConfig()
    config.hardware.device = "cpu"
    config.preference.enabled = True
    config.preference.algorithm = "simpo"
    config.training.batch_size = 1
    
    # Scale model parameters down to make CPU training extremely fast (nano model -> micro model)
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.num_experts = 2
    config.model.num_active_experts = 1
    config.model.titans_memory_dim = 16
    config.training.seq_len = 8
    
    # Mock offline datasets
    train_ds = PreferenceByteDataset(
        samples=[
            {"instruction": "Explain X", "chosen": "Y is X", "rejected": "Z is not X"},
            {"instruction": "Explain A", "chosen": "B is A", "rejected": "C is not A"},
            {"instruction": "Explain D", "chosen": "E is D", "rejected": "F is not D"},
            {"instruction": "Explain G", "chosen": "H is G", "rejected": "I is not G"},
            {"instruction": "Explain J", "chosen": "K is J", "rejected": "L is not J"}
        ],
        seq_len=8
    )
    
    # Patch autoregressive generation loops to avoid slow forward execution on CPU during unit testing
    with patch("evaluation.alignment_evaluator.AlignmentEvaluator._generate_sequence", return_value=b"dummy response"), \
         patch("evaluation.instruction_retention.InstructionRetentionEvaluator._generate", return_value=b"dummy instruction response"):
         
        res = run_preference_training(
            config=config,
            verification_level=1,
            train_ds_override=train_ds,
            val_ds_override=train_ds,
            seed=42
        )

    assert "final_loss" in res
    assert "checkpoint_dir" in res
