# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying Phase 6.3 namespace migrations, statistical methods, and verification chains."""

from __future__ import annotations

import os
import shutil
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from research.statistics import ResearchStatisticalValidator
from research.publication_manager import PublicationManager
from research.experiment_registry import ExperimentRegistry
from research.campaign_runner import CampaignRunner
from tests.provenance_helpers import seed_measured_experiment, seed_provenance_chain_experiments
import run_campaign
import replay_campaign


def test_phase_6_3_namespaces() -> None:
    """Verify that default namespaces have migrated to Phase 6.3."""
    # 1. run_campaign outputs
    assert "reports/phase_6_3/" in run_campaign.__doc__ or "Phase 6.3" in run_campaign.__doc__
    
    # 2. campaign_runner defaults
    runner = CampaignRunner(profile_name="verification")
    assert str(runner.output_dir) == str(Path("reports/phase_6_3"))
    
    # 3. replay_campaign defaults
    assert "reports/phase_6_3/" in replay_campaign.__doc__ or "experiments.db" in replay_campaign.__doc__


def test_compute_shapiro_wilk_scipy() -> None:
    """Verify Shapiro-Wilk SciPy code path when available."""
    validator = ResearchStatisticalValidator()
    # Generates a standard normal group
    normal_group = [0.1, -0.2, 0.05, 0.12, -0.05, 0.08, -0.15, 0.01, 0.03, -0.01]
    res = validator.compute_shapiro_wilk(normal_group)
    
    assert res["method"] == "scipy"
    assert "W" in res
    assert "p_value" in res
    assert isinstance(res["is_normal"], bool)


def test_compute_shapiro_wilk_fallback() -> None:
    """Verify Shapiro-Wilk fallback Royston approximation path when SciPy is mock-unavailable."""
    validator = ResearchStatisticalValidator()
    group_5 = [1.2, 1.5, 1.6, 1.8, 2.1]  # roughly normal
    
    with patch("scipy.stats.shapiro", side_effect=ImportError):
        res = validator.compute_shapiro_wilk(group_5)
        assert res["method"] == "royston_approx"
        assert 0.8 < res["W"] <= 1.0
        assert 0.0 < res["p_value"] <= 1.0
        assert isinstance(res["is_normal"], bool)


def test_apply_holm_bonferroni() -> None:
    """Verify Holm-Bonferroni correction outputs against expected textbook values."""
    validator = ResearchStatisticalValidator()
    # Family-wise raw p-values
    p_vals = {"H1": 0.01, "H2": 0.04, "H3": 0.20, "H4": 0.90}
    adjusted = validator.apply_holm_bonferroni(p_vals)
    
    # Raw order: H1 (0.01), H2 (0.04), H3 (0.20), H4 (0.90)
    # Adjusted calculations:
    # i=0: H1 -> 0.01 * 4 = 0.04
    # i=1: H2 -> max(0.04, 0.04 * 3) = 0.12
    # i=2: H3 -> max(0.12, 0.20 * 2) = 0.40
    # i=3: H4 -> max(0.40, 0.90 * 1) = 0.90
    assert abs(adjusted["H1"] - 0.04) < 1e-9
    assert abs(adjusted["H2"] - 0.12) < 1e-9
    assert abs(adjusted["H3"] - 0.40) < 1e-9
    assert abs(adjusted["H4"] - 0.90) < 1e-9


def test_apply_holm_bonferroni_monotone() -> None:
    """Verify adjusted p-values enforce non-decreasing monotonicity."""
    validator = ResearchStatisticalValidator()
    # Family-wise raw p-values with tie/inversion case
    p_vals = {"H1": 0.03, "H2": 0.02, "H3": 0.05}
    adjusted = validator.apply_holm_bonferroni(p_vals)
    
    # Raw order: H2 (0.02), H1 (0.03), H3 (0.05)
    # Adjusted:
    # H2 -> 0.02 * 3 = 0.06
    # H1 -> max(0.06, 0.03 * 2) = 0.06
    # H3 -> max(0.06, 0.05 * 1) = 0.06
    assert adjusted["H2"] == 0.06
    assert adjusted["H1"] == 0.06
    assert adjusted["H3"] == 0.06


def test_compute_cliffs_delta() -> None:
    """Verify Cliff's Delta effect size calculations for extreme and symmetric distributions."""
    validator = ResearchStatisticalValidator()
    
    # 1. Complete separation (B > A)
    group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    group_b = [6.0, 7.0, 8.0, 9.0, 10.0]
    res_sep = validator.compute_cliffs_delta(group_a, group_b)
    assert res_sep["delta"] == 1.0
    assert res_sep["magnitude"] == "large"
    
    # 2. Identical distributions
    res_ident = validator.compute_cliffs_delta(group_a, group_a)
    assert res_ident["delta"] == 0.0
    assert res_ident["magnitude"] == "negligible"


def test_generate_scientific_freeze(tmp_path: Path) -> None:
    """Verify that generate_scientific_freeze properly generates Phase_6_3_Freeze.md."""
    pub_mgr = PublicationManager(output_dir=str(tmp_path))
    
    # Test ValueError when parameters are missing or invalid
    with pytest.raises(TypeError):
        pub_mgr.generate_scientific_freeze()
        
    dataset_hashes = {"pretrain.bin": "hash_pretrain", "validation.bin": "hash_val"}
    prompt_hashes = {"HumanEval": "hash_he", "MBPP": "hash_mbpp"}
    benchmark_versions = {"HumanEval": "v1.0", "NeedleInHaystack": "v2.1"}
    
    freeze_file = pub_mgr.generate_scientific_freeze(
        git_sha="git_sha_123",
        dataset_hashes=dataset_hashes,
        prompt_hashes=prompt_hashes,
        benchmark_versions=benchmark_versions,
        campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER",
        experiment_count=35,
        archived_db_hash="hash_db_snapshot",
        replay_hash="hash_replay_final",
        phase_certificate_hash="hash_cert_final"
    )
    
    assert freeze_file.exists()
    assert freeze_file.name == "Phase_6_3_Freeze.md"
    
    content = freeze_file.read_text(encoding="utf-8")
    assert "IVERI CORE — Phase 6.3 Scientific Freeze Certificate" in content
    assert "git_commit" in content
    assert "dataset_hash[pretrain.bin]" in content
    assert "prompt_hash[HumanEval]" in content
    assert "benchmark_version[NeedleInHaystack]" in content
    assert "archived_db_hash" in content
    assert "replay_hash" in content
    assert "phase_certificate_hash" in content


def test_archive_database_integration(tmp_path: Path) -> None:
    """Verify that _archive_database copies experiments.db, hashes it, and logs artifact."""
    db_file = tmp_path / "experiments.db"
    # Create mock SQLite database
    registry = ExperimentRegistry(db_path=str(db_file))
    
    runner = CampaignRunner(profile_name="verification", db_path=str(db_file), output_dir=str(tmp_path))
    
    # Mock pre-requisites
    exp_id = "IVERI_Phase5_pretrain_Seed42_IVERI_Run001"
    registry.register_experiment(
        experiment_id=exp_id,
        purpose="unit test",
        hypothesis="H1",
        config_hash="conf_hash",
        git_sha="git_sha",
        git_branch="main",
        random_seed=42,
        tags=["iveri"]
    )
    
    # Run archiving
    archive_path = runner._archive_database(git_sha="git_sha_abc", run_uuid=exp_id)
    
    assert archive_path.exists()
    assert archive_path.name == "experiments_PHASE6_3_FINAL.db"
    assert archive_path.parent.name == "archives"
    
    # Check that artifact is registered
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT hash FROM checkpoints WHERE checkpoint_id = ?", ("db_archive_phase6_3",))
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert isinstance(row[0], str)


def test_extended_claim_provenance_chain_verification(tmp_path: Path) -> None:
    """Verify that _verify_claim_provenance_chain validates trace links correctly."""
    db_file = tmp_path / "experiments.db"
    output_dir = tmp_path / "reports"

    # Setup folders
    (output_dir / "statistics").mkdir(parents=True, exist_ok=True)
    (output_dir / "publication").mkdir(parents=True, exist_ok=True)

    # Generate empty reports
    hyp_report = output_dir / "statistics" / "Hypothesis_Report.md"
    evidence_index = output_dir / "publication" / "Evidence_Index.md"

    # Verify broken state initially
    res_broken = replay_campaign._verify_claim_provenance_chain(str(db_file), str(output_dir))
    assert res_broken is False

    registry = ExperimentRegistry(db_path=str(db_file))
    seed_provenance_chain_experiments(registry)

    # Write report files with references to all experiment IDs and hypotheses
    exp_mentions = "\n".join([f"IVERI_Phase5_pretrain_Seed42_IVERI_Run{i:03d}" for i in range(1, 11)])
    hyp_report.write_text(f"Hypothesis Report\n{exp_mentions}", encoding="utf-8")

    evidence_mentions = "\n".join([f"Hypothesis H{i} is SUPPORTED with p < 0.05" for i in range(1, 11)])
    evidence_index.write_text(f"Evidence Index\n{evidence_mentions}", encoding="utf-8")

    # Run verification again
    res_ok = replay_campaign._verify_claim_provenance_chain(str(db_file), str(output_dir))
    assert res_ok is True


def test_certificate_file_name_verification(tmp_path: Path) -> None:
    """Verify that PublicationManager generates Phase_6_3_Certificate.md filename."""
    db_file = tmp_path / "experiments.db"
    registry = ExperimentRegistry(db_path=str(db_file))
    seed_measured_experiment(registry, "exp_cert_01")
    pub_mgr = PublicationManager(registry=registry, output_dir=str(tmp_path))
    pub_mgr.generate_phase_certificate(campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER", total_runs=1)

    cert_file = tmp_path / "reviewer" / "Phase_6_3_Certificate.md"
    assert cert_file.exists()
    content = cert_file.read_text(encoding="utf-8")
    assert "Campaign Sign-Off Certificate — Phase 6.3" in content
    assert "Stage 8B / Phase 6.3" in content


def test_publication_blocks_non_measured_provenance(tmp_path: Path) -> None:
    """PublicationManager must fail closed when provenance is not MEASURED."""
    db_file = tmp_path / "experiments.db"
    registry = ExperimentRegistry(db_path=str(db_file))
    registry.register_experiment(
        experiment_id="exp_pending",
        purpose="test",
        hypothesis="H1",
        config_hash="conf",
        git_sha="a1b2c3d4e5f6789012345678abcdef9012345678",
        git_branch="main",
        random_seed=42,
        tags=["iveri"],
        provenance_label="UNKNOWN",
        status="PENDING",
    )
    pub_mgr = PublicationManager(registry=registry, output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="Publication blocked"):
        pub_mgr.generate_phase_certificate(campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER")


def test_publication_blocks_failed_runs(tmp_path: Path) -> None:
    """PublicationManager must fail closed when FAILED runs exist."""
    db_file = tmp_path / "experiments.db"
    registry = ExperimentRegistry(db_path=str(db_file))
    registry.register_experiment(
        experiment_id="exp_failed",
        purpose="test",
        hypothesis="H1",
        config_hash="conf",
        git_sha="a1b2c3d4e5f6789012345678abcdef9012345678",
        git_branch="main",
        random_seed=42,
        tags=["iveri"],
        provenance_label="UNKNOWN",
        status="FAILED",
    )
    pub_mgr = PublicationManager(registry=registry, output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="failed runs exist"):
        pub_mgr.generate_phase_certificate(campaign_id="IVERI_CAMPAIGN_2026_PHASE6_3_PAPER")
