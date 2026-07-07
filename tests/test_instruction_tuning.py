# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Test suite for Phase 3.2 SFT Instruction Tuning (Stage 2).

Verifies ConversationFormatter, LossMaskBuilder, SFTByteDataset,
InstructionDatasetLoader, SFTEvaluator, PromptSuite, ResponseInspector,
SFTCheckpointSelector, SFT Runner (run_sft), and full pipeline features.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import get_base_config
from evaluation.prompt_suite import PromptSuite, EvalPrompt
from evaluation.response_inspector import ResponseInspector
from evaluation.sft_evaluator import SFTEvaluator
from model.iveri_core import IVERIModel
from training.conversation_formatter import ConversationFormatter, FormatterConfig, TextSpan
from training.instruction_dataset import InstructionDatasetLoader
from training.loss_mask import LossMaskBuilder, MaskStrategy, apply_mask_to_loss
from training.model_selection import SFTCheckpointSelector
from training.sft_dataset import SFTByteDataset, make_sft_dataloader
from training.sft_runner import run_sft

# ── Mock objects for testing ────────────────────────────────────────────────


class MockSFTDataset(torch.utils.data.Dataset):
    """Mock dataset yielding random byte sequences of correct shape for SFT."""

    def __init__(self, num_samples: int = 10, seq_len: int = 16) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (seq_len - 1), y: (seq_len - 1), mask: (seq_len - 1)
        x = torch.randint(0, 256, (self.seq_len - 1,), dtype=torch.long)
        y = torch.randint(0, 256, (self.seq_len - 1,), dtype=torch.long)
        mask = torch.randint(0, 2, (self.seq_len - 1,), dtype=torch.bool)
        return x, y, mask


@pytest.fixture
def temp_dir():
    """Temporary directory for SFT verification."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def base_config():
    """IVERIConfig default configuration."""
    cfg = get_base_config(
        model={
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "mamba_ratio": 1,
            "num_experts": 2,
            "num_active_experts": 1,
            "max_recursion_depth": 2,
            "titans_memory_dim": 8,
        }
    )
    cfg.hardware.device = "cpu"
    cfg.hardware.mixed_precision = "fp32"
    cfg.training.seq_len = 16
    cfg.training.batch_size = 2
    cfg.training.gradient_accumulation = 1
    cfg.training.max_steps = 10
    cfg.logging.eval_every = 5
    cfg.logging.save_every = 5
    cfg.logging.log_every = 5
    cfg.instruction.enabled = True
    cfg.instruction.pretrained_checkpoint = ""
    return cfg


# ── 1. Conversation Formatter Tests ─────────────────────────────────────────


def test_conversation_formatter_alpaca():
    """Test single-turn Alpaca conversation formatting."""
    fmt = ConversationFormatter()
    sample = {"instruction": "Add 2+2", "input": "Now", "output": "4"}
    formatted = fmt.format_sample(sample)

    assert formatted.format_type == "alpaca"
    assert "### Instruction:\nAdd 2+2" in formatted.text
    assert "### Input:\nNow" in formatted.text
    assert "### Response:\n4" in formatted.text

    # Verify spans
    assert len(formatted.spans) == 2
    assert formatted.spans[0].role == "user"
    assert formatted.spans[1].role == "assistant"


def test_conversation_formatter_chat():
    """Test conversation formatting with user/assistant/system messages."""
    fmt = ConversationFormatter()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    formatted = fmt.format_messages(messages)

    assert formatted.format_type == "chat"
    assert "### System:\nYou are helpful." in formatted.text
    assert "### User:\nHello" in formatted.text
    assert "### Response:\nHi" in formatted.text

    # Verify spans
    assert len(formatted.spans) == 3
    assert formatted.spans[0].role == "system"
    assert formatted.spans[1].role == "user"
    assert formatted.spans[2].role == "assistant"


def test_conversation_formatter_multi_turn():
    """Test multi-turn dialog formatting."""
    fmt = ConversationFormatter()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Goodbye"},
    ]
    formatted = fmt.format_messages(messages)
    assert formatted.format_type == "multi_turn"
    assert formatted.num_turns == 2
    assert "Goodbye" in formatted.text


def test_conversation_formatter_max_turns():
    """Test that max_turns parameter trims dialogue turns correctly."""
    cfg = FormatterConfig(max_turns=1)
    fmt = ConversationFormatter(cfg)
    messages = [
        {"role": "system", "content": "helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Goodbye"},
    ]
    formatted = fmt.format_messages(messages)
    # Since max_turns = 1, it should keep system, first user and first assistant only.
    assert "Hi" in formatted.text
    assert "Goodbye" not in formatted.text


# ── 2. Loss Mask Builder Tests ──────────────────────────────────────────────


def test_loss_mask_builder():
    """Test building response-only and full loss masks."""
    builder = LossMaskBuilder(strategy=MaskStrategy.TRAIN_ONLY_ASSISTANT)
    spans = [
        TextSpan(0, 10, "user"),
        TextSpan(10, 20, "assistant"),
    ]
    seq_bytes = b"0123456789abcdefghij"
    result = builder.build(seq_bytes, spans=spans)

    # First 10 bytes (user) should be masked (False), next 10 bytes (assistant) unmasked (True)
    assert result.mask[:10].sum().item() == 0
    assert result.mask[10:20].sum().item() == 10

    # Test full mask strategy
    builder_full = LossMaskBuilder(strategy=MaskStrategy.TRAIN_ENTIRE_SEQUENCE)
    result_full = builder_full.build(seq_bytes, spans=spans)
    assert result_full.mask.sum().item() == 20


def test_loss_mask_padding():
    """Test that padding bytes are automatically masked."""
    builder = LossMaskBuilder(strategy=MaskStrategy.TRAIN_ENTIRE_SEQUENCE, pad_byte=0)
    # Sequence with padding bytes at the end
    seq = bytes([1, 2, 3, 0, 0])
    result = builder.build(seq)
    assert result.mask.tolist() == [True, True, True, False, False]


# ── 3. SFT Byte Dataset Tests ───────────────────────────────────────────────


def test_sft_byte_dataset():
    """Test SFTByteDataset tokenization and autoregressive shift."""
    samples = [
        {"instruction": "Add 2+2", "output": "The output is 4."},
        {"instruction": "Add 3+3", "output": "The output is 6."},
    ]
    ds = SFTByteDataset(samples, seq_len=16, train_on_prompt=False)
    assert len(ds) == 2

    x, y, mask = ds[0]
    # Check shapes
    assert x.shape == (15,)
    assert y.shape == (15,)
    assert mask.shape == (15,)

    # autoregressive shift: y should be shifted x
    assert torch.equal(x[1:], y[:-1])


# ── 4. SFT Dataset Loader Validation ────────────────────────────────────────


def test_instruction_dataset_loader_license_and_validation(temp_dir, base_config):
    """Test InstructionDatasetLoader validation logic."""
    dataset_name = "test_sft_data"
    processed_dir = temp_dir / "stage2" / dataset_name
    processed_dir.mkdir(parents=True)

    # Create VERSION.json
    with open(processed_dir / "VERSION.json", "w") as f:
        json.dump({"stage": 2, "version_id": "1.0.0"}, f)

    # Create train.jsonl
    with open(processed_dir / "train.jsonl", "w") as f:
        f.write(json.dumps({"instruction": "What is 2+2?", "output": "The answer is four."}) + "\n")

    loader = InstructionDatasetLoader(base_config)
    loader._processed_base = temp_dir

    # Mock DataRegistry entry to check license validation
    class MockEntry:
        name = "test_sft_data"
        license_id = "Apache-2.0"
        stage = 2

    with patch.object(loader, "_get_dataset_entry", return_value=MockEntry()):
        ds = loader.load(dataset_name, split="train")
        assert len(ds) == 1
        x, y, mask = ds[0]
        assert x.shape == (15,)


# ── 5. Response Inspector & Prompt Suite Tests ──────────────────────────────


def test_response_inspector():
    """Test ResponseInspector for quality assessment."""
    inspector = ResponseInspector()

    # Good response
    good = b"Paris is the capital of France."
    res_good = inspector.inspect_bytes(good)
    assert res_good.is_valid
    assert not res_good.is_collapsed
    assert not res_good.has_excessive_loops

    # Collapsed/empty/repetition response
    bad = b"a" * 50
    res_bad = inspector.inspect_bytes(bad)
    assert not res_bad.is_valid
    assert "collapse" in res_bad.issues or "repetition" in res_bad.issues


def test_prompt_suite():
    """Test PromptSuite deterministic categories and format."""
    suite = PromptSuite()
    assert len(suite) >= 30
    assert "coding" in suite.get_categories()

    prompts = suite.get_category("coding")
    assert len(prompts) > 0
    assert isinstance(prompts[0], EvalPrompt)
    assert prompts[0].instruction


# ── 6. SFT Evaluator Tests ──────────────────────────────────────────────────


def test_sft_evaluator(base_config):
    """Test SFTEvaluator compute metrics and run generation."""
    model = IVERIModel(base_config)
    evaluator_base = MagicMock()
    evaluator_base.model = model
    evaluator_base.device = "cpu"
    evaluator_base.precision_handler = MagicMock()

    sft_eval = SFTEvaluator(evaluator_base, base_config)

    # Mock DataLoader
    mock_ds = MockSFTDataset(num_samples=2, seq_len=16)
    val_loader = DataLoader(mock_ds, batch_size=2)

    # Test evaluate_sft metrics
    metrics = sft_eval.evaluate_sft(val_loader)
    assert "sft/val_loss" in metrics
    assert "sft/perplexity" in metrics

    # Test evaluate_prompt_suite
    suite = PromptSuite(prompts=[
        {"prompt_id": "test_01", "category": "coding", "instruction": "Print Hello", "expected_keywords": ["Hello"]}
    ])
    res = sft_eval.evaluate_prompt_suite(suite, max_new_bytes=4)
    assert "avg_quality_score" in res
    assert len(res["per_prompt"]) == 1


# ── 7. Checkpoint Selection & Experiment Manager Extension Tests ────────────


def test_checkpoint_selection(temp_dir):
    """Test SFTCheckpointSelector metrics and ranking."""
    selector = SFTCheckpointSelector(log_dir=temp_dir)
    selector.register_checkpoint(
        path=temp_dir / "model_50.pt",
        step=50,
        train_loss=1.5,
        val_loss=1.2,
        perplexity=3.32,
        response_quality_score=0.8,
    )
    selector.register_checkpoint(
        path=temp_dir / "model_100.pt",
        step=100,
        train_loss=1.0,
        val_loss=1.4,
        perplexity=4.05,
        response_quality_score=0.9,
    )

    best_loss = selector.get_best_checkpoint("val_loss")
    assert best_loss["step"] == 50

    best_quality = selector.get_best_checkpoint("response_quality_score")
    assert best_quality["step"] == 100

    best_joint = selector.get_best_sft_checkpoint()
    assert best_joint is not None


# ── 8. E2E SFT Verification Run (run_sft) ───────────────────────────────────


def test_sft_runner_e2e(temp_dir, base_config):
    """Verify run_sft e2e with mock datasets."""
    # Write mock config paths
    base_config.logging.save_dir = str(temp_dir)
    base_config.logging.mode = "disabled"

    # Mock instruction dataset loaders
    mock_ds = MockSFTDataset(num_samples=4, seq_len=16)

    # Run SFT verification Level 1 (20 steps)
    results = run_sft(
        config=base_config,
        verification_level=1,
        dataset_name="mock_sft_data",
        train_ds_override=mock_ds,
        val_ds_override=mock_ds,
        seed=123,
    )

    assert results["final_loss"] > 0
    assert "checkpoint_dir" in results
    assert "generation_results" in results
