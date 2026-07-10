# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive test suite for Phase 3.3 Coding Specialization (Stage 3A).

Verifies CodingConfig, CodingDatasetLoader, CodeFormatter, CodingCurriculum,
CodeInspector, InstructionRetentionEvaluator, HumanEvalBenchmark, MBPPBenchmark,
CodeExecutor, CodeQualityAnalyzer, SecurityScanner, ContaminationChecker,
CodingCheckpointSelector, and coding runner (run_coding).
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

from configs.base_config import get_base_config
from configs.coding_config import CodingConfig
from evaluation.coding_prompt_suite import CodingPromptSuite, CodeEvalPrompt
from evaluation.code_inspector import CodeInspector
from evaluation.instruction_retention import InstructionRetentionEvaluator
from evaluation.code_execution import CodeExecutor
from evaluation.code_quality_analyzer import CodeQualityAnalyzer
from evaluation.security_scanner import SecurityScanner
from evaluation.contamination_checker import ContaminationChecker
from model.iveri_core import IVERIModel
from training.coding_dataset import CodingDatasetLoader
from training.code_formatter import CodeFormatter, CodeFormatterConfig
from training.coding_curriculum import CodingCurriculum
from training.model_selection import CodingCheckpointSelector
from training.coding_runner import run_coding


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def base_config():
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
    cfg.coding.enabled = True
    cfg.coding.sft_checkpoint = ""
    return cfg


# ── 1. Config Tests ────────────────────────────────────────────────────────


def test_coding_config_defaults():
    cfg = CodingConfig()
    assert not cfg.enabled
    assert "nemotron_competitive" in cfg.datasets
    assert cfg.generation_temperature == 0.2
    assert cfg.instruction_retention_enabled
    assert cfg.instruction_retention_threshold == 0.15


def test_coding_config_validation():
    from core.exceptions import ConfigError
    with pytest.raises(ConfigError):
        CodingConfig(curriculum_stages=4)
    with pytest.raises(ConfigError):
        CodingConfig(generation_temperature=-0.5)


def test_coding_config_in_iveri_config(base_config):
    assert hasattr(base_config, "coding")
    assert isinstance(base_config.coding, CodingConfig)


def test_coding_config_from_dict():
    from configs.base_config import IVERIConfig
    raw = {"coding": {"enabled": True, "generation_temperature": 0.5}}
    cfg = IVERIConfig.from_dict(raw)
    assert cfg.coding.enabled
    assert cfg.coding.generation_temperature == 0.5


# ── 2. Formatter Tests ──────────────────────────────────────────────────────


def test_code_formatter_pretrain():
    fmt = CodeFormatter()
    raw = {"content": "print('hello')", "language": "python"}
    b, spans = fmt.format_to_bytes(raw)
    assert len(b) > 0
    assert len(spans) == 1
    assert spans[0].role == "assistant"


def test_code_formatter_sft_alpaca():
    fmt = CodeFormatter()
    raw = {"instruction": "Write sum", "output": "def sum(): return 1"}
    b, spans = fmt.format_to_bytes(raw)
    assert len(b) > 0
    # Expected roles in spans: user, assistant, system or prefixes
    roles = {s.role for s in spans}
    assert "assistant" in roles


def test_code_formatter_language_header():
    cfg = CodeFormatterConfig(include_language_header=True)
    fmt = CodeFormatter(cfg)
    raw = {"instruction": "Write sum", "output": "def sum(): return 1", "language": "python"}
    b, _ = fmt.format_to_bytes(raw)
    text = b.decode("utf-8")
    assert "### Language: python" in text


# ── 3. Curriculum Tests ─────────────────────────────────────────────────────


def test_coding_curriculum_stages():
    curr = CodingCurriculum(num_stages=3)
    assert curr.get_stage_index(0, 100) == 0
    assert curr.get_stage_index(40, 100) == 1
    assert curr.get_stage_index(80, 100) == 2


def test_coding_curriculum_dataset_weights():
    curr = CodingCurriculum(num_stages=3)
    w0 = curr.get_active_datasets(0, 100)
    w1 = curr.get_active_datasets(40, 100)
    w2 = curr.get_active_datasets(80, 100)
    assert abs(sum(w0.values()) - 1.0) < 1e-3
    assert abs(sum(w1.values()) - 1.0) < 1e-3
    assert abs(sum(w2.values()) - 1.0) < 1e-3


# ── 4. Dataset Loader Tests ─────────────────────────────────────────────────


def test_coding_dataset_loader_mock(base_config):
    loader = CodingDatasetLoader(base_config)
    samples = [{"instruction": "Write reverse", "output": "s[::-1]"}]
    ds = loader.load_mock(samples)
    assert len(ds) == 1
    x, y, mask = ds[0]
    assert x.shape == (511,)
    assert y.shape == (511,)


def test_coding_dataset_loader_format_type(base_config):
    loader = CodingDatasetLoader(base_config)
    assert loader.get_format_type("the_stack_v2_deep") == "pretrain"
    assert loader.get_format_type("leetcode") == "sft"


# ── 5. Code Inspector Tests ─────────────────────────────────────────────────


def test_code_inspector_syntax_valid():
    insp = CodeInspector()
    code = b"### Language: python\ndef hello():\n    return 42\n"
    res = insp.inspect_bytes(code)
    assert res.syntax_valid is True
    assert res.language_detected == "python"


def test_code_inspector_syntax_invalid():
    insp = CodeInspector()
    code = b"### Language: python\ndef hello(\n"
    res = insp.inspect_bytes(code)
    assert res.syntax_valid is False


def test_code_inspector_non_python_skip():
    insp = CodeInspector()
    code = b"### Language: rust\nfn hello() {\n"
    res = insp.inspect_bytes(code)
    # Non-Python without tree-sitter returns None (no penalty)
    assert res.syntax_valid is None


# ── 6. Instruction Retention Tests ──────────────────────────────────────────


def test_instruction_retention_evaluator(base_config):
    model = IVERIModel(base_config)
    evaluator = InstructionRetentionEvaluator(
        model=model,
        config=base_config,
        baseline_quality_score=0.8,
    )
    # Patch SFT PromptSuite and generation for speed
    with patch.object(evaluator, "_generate", return_value=b"hello response"):
        res = evaluator.evaluate(step=0)
        assert "instruction/pass_rate" in res
        assert "instruction/quality_score" in res
        assert "instruction/quality_delta" in res


# ── 7. Code Execution Tests ─────────────────────────────────────────────────


def test_code_executor_compile_success():
    exc = CodeExecutor()
    res = exc.execute("def func():\n    return 1")
    assert res.compile_success


def test_code_executor_compile_fail():
    exc = CodeExecutor()
    res = exc.execute("def func(")
    assert not res.compile_success


def test_code_executor_runtime_error():
    exc = CodeExecutor()
    # Runs in separate process, will crash on division by zero
    res = exc.execute("1 / 0")
    assert res.compile_success
    assert not res.execution_success
    assert res.runtime_error


# ── 8. Code Quality Analyzer Tests ──────────────────────────────────────────


def test_code_quality_analyzer_python():
    analyzer = CodeQualityAnalyzer()
    code = (
        "def func(x):\n"
        "    if x > 0:\n"
        "        return x\n"
        "    return 0\n"
    )
    res = analyzer.analyze(code, "python")
    assert res.function_count == 1
    assert res.cyclomatic_complexity >= 2.0


def test_code_quality_analyzer_heuristic():
    analyzer = CodeQualityAnalyzer()
    code = (
        "fn main() {\n"
        "    // print hello\n"
        "    println!(\"hello\");\n"
        "}\n"
    )
    res = analyzer.analyze(code, "rust")
    assert res.comment_ratio > 0.0
    assert res.analysis_method == "heuristic"


# ── 9. Security Scanner Tests ───────────────────────────────────────────────


def test_security_scanner_eval():
    scanner = SecurityScanner()
    code = "x = eval(input())"
    res = scanner.scan(code)
    assert res.is_flagged
    assert "eval_usage" in res.flagged_patterns
    assert res.risk_level == "medium"


def test_security_scanner_clean():
    scanner = SecurityScanner()
    code = "def clean():\n    return 42"
    res = scanner.scan(code)
    assert not res.is_flagged
    assert res.risk_level == "none"


# ── 10. Contamination Checker Tests ─────────────────────────────────────────


def test_contamination_checker(temp_dir):
    checker = ContaminationChecker()
    bench = [
        {"prompt_id": "test_1", "instruction": "def reverse_string(s):\n    return s[::-1]"}
    ]
    # No contamination matching
    res = checker.check(bench, temp_dir)
    assert res.clean
    assert res.contaminated_count == 0


# ── 11. Checkpoint Selector Tests ───────────────────────────────────────────


def test_coding_checkpoint_selector(temp_dir):
    sel = CodingCheckpointSelector(log_dir=temp_dir)
    # Register clean checkpoint
    sel.register_checkpoint(
        path="chk_1.pt",
        step=50,
        train_loss=1.0,
        val_loss=1.1,
        perplexity=3.0,
        code_quality_score=0.9,
        syntax_valid_ratio=1.0,
        instruction_retention_ok=True,
    )
    # Register regression checkpoint
    sel.register_checkpoint(
        path="chk_2.pt",
        step=100,
        train_loss=0.8,
        val_loss=0.9,
        perplexity=2.5,
        code_quality_score=0.95,
        syntax_valid_ratio=1.0,
        instruction_retention_ok=False,  # fails retention
    )
    best = sel.get_best_coding_checkpoint()
    # Should select chk_1 since chk_2 has regression
    assert best["step"] == 50


# ── 12. Runner Smoke Test ───────────────────────────────────────────────────


def test_run_coding_mock(base_config, monkeypatch):
    """Smoke-test run_coding() with mock datasets.

    Sets IVERI_OFFLINE=1 so HumanEval/MBPP benchmarks skip network calls
    and return stub results immediately.
    """
    monkeypatch.setenv("IVERI_OFFLINE", "1")
    base_config.hardware.num_workers = 0
    base_config.logging.mode = "disabled"

    class _MockCodingDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = torch.randint(0, 256, (15,), dtype=torch.long)
            y = torch.randint(0, 256, (15,), dtype=torch.long)
            mask = torch.ones(15, dtype=torch.bool)
            return x, y, mask

    train_ds = _MockCodingDataset()
    val_ds = _MockCodingDataset()

    results = run_coding(
        config=base_config,
        verification_level=1,  # 20 steps
        train_ds_override=train_ds,
        val_ds_override=val_ds,
    )
    assert "final_loss" in results
    assert "final_val_loss" in results
    assert "curriculum_stage_history" in results

