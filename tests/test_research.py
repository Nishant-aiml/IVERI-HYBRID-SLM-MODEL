# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for Phase 3.5 Research Validation and Benchmarking.

Verifies baseline creation, checkpointing, ablations, profiling, calibration,
statistics, scorecards, and paper helper artifact generators.
"""

from __future__ import annotations

import tempfile
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import IVERIConfig
from configs.research_config import ResearchConfig
from core.exceptions import ConfigError
from research.baselines import BaselineManager, BaselineMamba2, BaselineHybrid
from research.checkpoint_manager import BaselineCheckpointManager
from research.ablation import AblationSuite
from research.experiment_runner import ExperimentRunner
from research.multi_seed import MultiSeedRunner
from research.flops import FlopProfiler
from research.profiler import MemoryProfiler
from research.energy_profiler import EnergyProfiler
from research.calibration import ConfidenceCalibrator
from research.benchmark_research import ResearchBenchmarkRunner
from research.benchmark_engineering import EngineeringBenchmarkRunner
from research.scaling import ScalingAnalyzer
from research.statistics import ResearchStatisticalValidator
from research.claim_validator import ClaimValidator
from research.hypothesis import ResearchHypothesisEngine
from research.paper_figures import PaperFigureGenerator
from research.paper_tables import PaperTableGenerator
from research.paper_summary import PaperSummaryGenerator
from research.artifacts import ResearchArtifactsManager


# Helper dummy model matching interfaces
class DummyModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(256, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 256)
        self.param = nn.Parameter(torch.ones(1, hidden_dim))

    def forward(self, x, **kwargs):
        # shape: (B, S) -> (B, S, 256)
        embeds = self.embedding(x)
        logits = self.lm_head(embeds)
        return {"logits": logits, "loss": logits.mean()}


# ── Test 1-3: Research Config ─────────────────────────────────────────
def test_research_config_defaults():
    cfg = ResearchConfig()
    assert not cfg.enabled
    assert len(cfg.random_seeds) == 5
    assert cfg.num_runs == 5
    assert cfg.report_directory == "reports/phase_3_5/"


def test_research_config_validation():
    with pytest.raises(ConfigError):
        ResearchConfig(random_seeds=[])
    with pytest.raises(ConfigError):
        ResearchConfig(num_runs=0)
    with pytest.raises(ConfigError):
        ResearchConfig(confidence_interval=1.2)


def test_research_config_base_integration():
    config = IVERIConfig()
    assert hasattr(config, "research")
    assert not config.research.enabled


# ── Test 4-6: Baseline Manager ────────────────────────────────────────
def test_baseline_manager_transformer():
    config = IVERIConfig()
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    manager = BaselineManager(config)
    model = manager.build_transformer_baseline()
    assert isinstance(model, nn.Module)
    assert model.hidden_dim == 16


def test_baseline_manager_mamba():
    config = IVERIConfig()
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    manager = BaselineManager(config)
    model = manager.build_mamba_baseline()
    assert isinstance(model, BaselineMamba2)
    assert model.hidden_dim == 16


def test_baseline_manager_hybrid():
    config = IVERIConfig()
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    manager = BaselineManager(config)
    model = manager.build_hybrid_baseline()
    assert isinstance(model, BaselineHybrid)
    assert model.hidden_dim == 16


# ── Test 7-9: Baseline Checkpoint Manager ──────────────────────────────
def test_checkpoint_manager_param_count():
    config = IVERIConfig()
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = Path(tmp_dir) / "registry.json"
        mgr = BaselineCheckpointManager(config, registry_path=str(registry))
        count = mgr.count_parameters(model)
        assert count > 0


def test_checkpoint_manager_save_load():
    config = IVERIConfig()
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = Path(tmp_dir) / "registry.json"
        ckpt_path = Path(tmp_dir) / "baseline.pt"
        mgr = BaselineCheckpointManager(config, registry_path=str(registry))
        
        sha = mgr.save_checkpoint(model, ckpt_path, step=10, metrics={"loss": 2.0})
        assert sha != ""
        assert ckpt_path.exists()

        loaded_info = mgr.load_checkpoint(model, ckpt_path)
        assert loaded_info["step"] == 10
        assert loaded_info["metrics"]["loss"] == 2.0


def test_checkpoint_manager_parity():
    config = IVERIConfig()
    model_a = DummyModel(hidden_dim=32)
    model_b = DummyModel(hidden_dim=32)
    model_c = DummyModel(hidden_dim=64)

    mgr = BaselineCheckpointManager(config)
    assert mgr.verify_parity(model_a, model_b)
    assert not mgr.verify_parity(model_a, model_c)


# ── Test 10-12: Ablation Suite ────────────────────────────────────────
def test_ablation_suite_full():
    config = IVERIConfig()
    suite = AblationSuite(config)
    model = suite.get_ablated_model("full")
    assert isinstance(model, nn.Module)


def test_ablation_suite_components():
    config = IVERIConfig()
    config.hardware.device = "cpu"
    suite = AblationSuite(config)

    model_titans = suite.get_ablated_model("no_titans")
    assert model_titans.config.model.use_titans is False
    assert model_titans.backbone.titans is None

    model_mor = suite.get_ablated_model("no_mor")
    assert model_mor.config.model.use_mor is False

    model_moe = suite.get_ablated_model("no_moe")
    assert model_moe.config.model.use_moe is False

    model_blt = suite.get_ablated_model("no_blt")
    assert model_blt.config.model.use_blt is False
    assert not hasattr(model_blt, "entropy_model")

    model_ent = suite.get_ablated_model("no_entropy_routing")
    assert model_ent.config.model.use_entropy_routing is False


def test_ablation_suite_evaluation():
    config = IVERIConfig()
    suite = AblationSuite(config)
    
    def dummy_eval(model):
        return {"eval_loss": 1.5}

    res = suite.run_ablation_evaluation("no_moe", dummy_eval)
    assert res["eval_loss"] == 1.5


# ── Test 13-15: Runners ───────────────────────────────────────────────
def test_experiment_runner_run():
    config = IVERIConfig()
    config.hardware.device = "cpu"
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.training.learning_rate = 1e-4

    model = DummyModel(hidden_dim=16)
    inputs = torch.randint(0, 256, (4, 8), dtype=torch.long)
    dataset = TensorDataset(inputs)
    loader = DataLoader(dataset, batch_size=2)

    runner = ExperimentRunner(config)
    metrics = runner.run_experiment(model, loader, val_loader=loader, max_steps=2)
    assert "train_loss" in metrics
    assert "perplexity" in metrics
    assert metrics["steps"] == 2


def test_multi_seed_runner():
    config = IVERIConfig()
    config.hardware.device = "cpu"
    config.research.random_seeds = [42, 123]
    config.model.hidden_dim = 16
    config.model.num_layers = 1
    config.model.num_heads = 2

    # builder lambda
    builder = lambda: DummyModel(hidden_dim=16)
    
    inputs = torch.randint(0, 256, (4, 8), dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs), batch_size=2)

    runner = MultiSeedRunner(config)
    stats = runner.run_multi_seed(builder, loader, val_loader=loader, max_steps=2)
    
    assert "train_loss" in stats
    assert "mean" in stats["train_loss"]
    assert "std" in stats["train_loss"]


def test_multi_seed_ci_boundaries():
    config = IVERIConfig()
    config.hardware.device = "cpu"
    config.research.random_seeds = [1, 2]
    runner = MultiSeedRunner(config)
    
    builder = lambda: DummyModel(hidden_dim=16)
    loader = DataLoader(TensorDataset(torch.randint(0, 256, (4, 8), dtype=torch.long)), batch_size=2)
    
    stats = runner.run_multi_seed(builder, loader, val_loader=loader, max_steps=1)
    ci_lower = stats["train_loss"]["ci_lower"]
    ci_upper = stats["train_loss"]["ci_upper"]
    assert ci_lower <= ci_upper


# ── Test 16-18: FLOP Profiler ──────────────────────────────────────────
def test_flop_profiler_attention():
    config = IVERIConfig()
    config.model.hidden_dim = 64
    config.model.num_heads = 4
    config.training.seq_len = 16
    profiler = FlopProfiler(config)
    flops = profiler.estimate_attention_flops()
    assert flops > 0.0


def test_flop_profiler_mamba():
    config = IVERIConfig()
    config.model.hidden_dim = 64
    config.training.seq_len = 16
    profiler = FlopProfiler(config)
    flops = profiler.estimate_mamba_flops()
    assert flops > 0.0


def test_flop_profiler_moe_titans_blt():
    config = IVERIConfig()
    config.model.hidden_dim = 64
    config.model.num_heads = 4
    config.training.seq_len = 16
    config.model.num_experts = 4
    config.model.num_active_experts = 2
    config.model.titans_memory_dim = 32

    profiler = FlopProfiler(config)
    assert profiler.estimate_moe_flops() > 0.0
    assert profiler.estimate_titans_flops() > 0.0
    assert profiler.estimate_blt_flops() > 0.0
    assert profiler.calculate_forward_flops() > 0.0
    assert profiler.calculate_total_training_flops(1000) > 0.0


# ── Test 19-22: System and Subsystem Profiler ─────────────────────────
def test_memory_profiler_ram_vram():
    config = IVERIConfig()
    prof = MemoryProfiler(config)
    mem_info = prof.get_system_memory_info()
    assert "cpu_ram_rss_mb" in mem_info
    assert "gpu_vram_allocated_mb" in mem_info


def test_memory_profiler_subsystem_diagnostics():
    config = IVERIConfig()
    prof = MemoryProfiler(config)
    model = DummyModel()
    inputs = torch.randint(0, 256, (1, 8), dtype=torch.long)
    diag = prof.profile_subsystem_diagnostics(model, inputs)
    assert "router" in diag
    assert "titans" in diag
    assert "blt" in diag
    assert not diag["is_flagged"]


def test_memory_profiler_latency():
    config = IVERIConfig()
    prof = MemoryProfiler(config)
    model = DummyModel()
    lat = prof.profile_decoding_latency(model, b"hello")
    assert "time_to_first_token_sec" in lat
    assert "decode_speed_tokens_per_sec" in lat


def test_memory_profiler_throughput_context():
    config = IVERIConfig()
    prof = MemoryProfiler(config)
    model = DummyModel()
    res = prof.profile_throughput_vs_context(model, context_lengths=[32, 64])
    assert 32 in res
    assert 64 in res


# ── Test 23-24: Energy Profiler ───────────────────────────────────────
def test_energy_profiler_watts():
    prof = EnergyProfiler()
    watts = prof._get_gpu_power_watts()
    assert watts > 0.0


def test_energy_profiler_session():
    prof = EnergyProfiler()
    prof.start_session()
    prof.poll()
    metrics = prof.stop_session_and_compute(total_tokens=100)
    assert "energy_per_token_joules" in metrics
    assert "tokens_per_joule" in metrics


# ── Test 25-26: Calibration ───────────────────────────────────────────
def test_calibration_ece_brier():
    cal = ConfidenceCalibrator(num_bins=5)
    # 2 samples, 3 classes
    logits = torch.tensor([
        [2.0, 0.5, 0.1],
        [0.1, 3.0, 0.2]
    ])
    labels = torch.tensor([0, 1]) # 100% correct predictions
    res = cal.compute_calibration_metrics(logits, labels)
    
    assert "expected_calibration_error" in res
    assert "brier_score" in res
    assert "negative_log_likelihood" in res
    assert len(res["reliability_diagram"]) == 5


def test_calibration_brier_limit():
    cal = ConfidenceCalibrator(num_bins=5)
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    res = cal.compute_calibration_metrics(logits, labels)
    assert res["brier_score"] >= 0.0


# ── Test 27-28: Benchmarks ────────────────────────────────────────────
def test_research_benchmark_needle():
    config = IVERIConfig()
    runner = ResearchBenchmarkRunner(config)
    model = DummyModel()
    res = runner.run_needle_in_a_haystack(model, context_len=100, needle_pos=50)
    assert "success" in res
    assert "predicted_byte" in res


def test_engineering_benchmark_logic():
    config = IVERIConfig()
    runner = EngineeringBenchmarkRunner(config)
    model = DummyModel()
    res = runner.run_logic_gsm8k(model)
    assert "gsm8k_accuracy" in res


# ── Test 29-30: Scaling Law ───────────────────────────────────────────
def test_scaling_law_fit():
    analyzer = ScalingAnalyzer()
    x = [10.0, 50.0, 100.0]
    y = [2.5, 1.8, 1.5]
    fit = analyzer.fit_power_law(x, y)
    assert fit["a"] > 0
    assert fit["b"] > 0
    assert 0.0 <= fit["r_squared"] <= 1.0


def test_scaling_law_validation():
    analyzer = ScalingAnalyzer()
    x = [10.0, 50.0, 100.0]
    y = [2.5, 1.8, 1.5]
    fit = analyzer.fit_power_law(x, y)
    val = analyzer.validate_predictions(x, y, fit)
    assert val["rmse"] >= 0.0
    assert val["mape"] >= 0.0


# ── Test 31-33: Statistics ────────────────────────────────────────────
def test_statistics_t_test():
    val = ResearchStatisticalValidator()
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.1, 2.2, 3.1, 4.3, 5.2]
    res = val.compute_paired_t_test(a, b)
    assert "t_statistic" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_statistics_wilcoxon():
    val = ResearchStatisticalValidator()
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.1, 2.2, 3.1, 4.3, 5.2]
    res = val.compute_wilcoxon_signed_rank(a, b)
    assert "w_statistic" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_statistics_cohens_d_and_bootstrap():
    val = ResearchStatisticalValidator()
    a = [1.0, 2.0, 3.0]
    b = [2.0, 3.0, 4.0]
    d = val.compute_cohens_d(a, b)
    assert d > 0.0

    ci = val.compute_bootstrap_confidence_interval(a, b, num_resamples=50)
    assert ci[0] <= ci[1]


# ── Test 34-35: Scorecard & Claims ────────────────────────────────────
def test_claim_validator_classification():
    cv = ClaimValidator()
    assert cv.validate_claim("Test", p_value=0.01, delta_pct=0.05, num_seeds=5) == "SUPPORTED"
    assert cv.validate_claim("Test", p_value=0.08, delta_pct=0.03, num_seeds=3) == "LIKELY"
    assert cv.validate_claim("Test", p_value=0.25, delta_pct=0.01, num_seeds=5) == "HYPOTHESIS"
    assert cv.validate_claim("Test", p_value=None, delta_pct=0.05, num_seeds=5) == "UNVERIFIED"
    assert cv.validate_claim("Test", p_value=0.15, delta_pct=-0.08, num_seeds=5) == "REFUTED"


def test_claim_validator_scorecards():
    cv = ClaimValidator()
    rep_score = cv.calculate_reproducibility_score(
        git_sha="abcdef", config_hash="12345", seed_count=5, checksums_ok=True, env_captured=True
    )
    assert rep_score == 100.0

    integrity_score = cv.calculate_research_integrity_score(
        baseline_coverage_ok=True, completed_ablations=4, total_ablations=4,
        calibration_completed=True, seeds_run=5, statistical_significance_run=True
    )
    assert integrity_score == 100.0


# ── Test 36: Hypothesis Engine ────────────────────────────────────────
def test_hypothesis_engine_evaluation():
    engine = ResearchHypothesisEngine()
    res = engine.evaluate_hypothesis("H1", delta_pct=0.12, p_value=0.01, num_seeds=5)
    assert res["hypothesis_label"] == "H1"
    assert res["outcome"] == "SUPPORTED"

    res_refuted = engine.evaluate_hypothesis("H1", delta_pct=-0.05, p_value=0.20, num_seeds=5)
    assert res_refuted["outcome"] == "REFUTED"


# ── Test 37-39: Paper Helpers & Artifacts ──────────────────────────────
def test_paper_helpers_latex_summary():
    gen_tab = PaperTableGenerator()
    gen_sum = PaperSummaryGenerator()

    latex = gen_tab.generate_benchmark_table({"humaneval": 0.8}, {"humaneval": 0.5}, {"humaneval": 0.6}, {"humaneval": 0.7})
    assert "Vanilla Transformer" in latex

    summary = gen_sum.generate_discussion_summary([], 98.0, 95.0)
    assert "Academic Discussion" in summary


def test_paper_figures_export():
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = PaperFigureGenerator(output_dir=tmp_dir)
        paths = gen.plot_loss_curves([1, 2], [1.0, 0.8], [1.2, 1.0], [1.1, 0.9], [1.15, 0.95])
        assert len(paths) == 3
        assert paths[0].exists()


def test_artifacts_reproducibility_zip():
    config = IVERIConfig()
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = ResearchArtifactsManager(config, output_dir=tmp_dir)
        zip_path = mgr.export_reproducibility_package(
            experiment_metrics={"accuracy": 0.9},
            fig_paths=[]
        )
        assert zip_path.exists()


# ── Test 40: Package Init Exports ─────────────────────────────────────
def test_package_exports():
    import research
    assert hasattr(research, "BaselineManager")
    assert hasattr(research, "BaselineCheckpointManager")
    assert hasattr(research, "ConfidenceCalibrator")
    assert hasattr(research, "ResearchHypothesisEngine")
