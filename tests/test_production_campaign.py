# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration test suite for Phase 5.0 production validation campaign."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from configs.base_config import get_base_config
from research.campaign_runner import CampaignRunner
from research.experiment_registry import ExperimentRegistry
from research.campaign_config import CampaignConfig
from research.execution_backend import ExecutionBackend
from research.cost_estimator import CostEstimator
from research.campaign_dataset_validator import CampaignDatasetValidator
from research.campaign_lock import CampaignLock
from research.campaign_health_monitor import CampaignHealthMonitor
from research.experiment_manifest import ExperimentManifestGenerator
from research.publication_manager import PublicationManager
from research.external_eval import ExternalModelEvaluator
from research.golden import GoldenCheckpointManager
from research.benchmark_integrity import BenchmarkIntegrityFramework
from tests.provenance_helpers import BENCHMARK_REGISTRY_HASH, CONFIG_HASH, GIT_SHA, seed_measured_experiment


class TestProductionCampaign(unittest.TestCase):
    """35 comprehensive tests verifying campaign orchestration components."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp_dir) / "test_campaign.db")
        self.registry = ExperimentRegistry(db_path=self.db_path)
        self.config = get_base_config()

    def tearDown(self) -> None:
        # Clean up temporary folders
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # --- 1-4. Campaign Config Profile Tests ---
    def test_campaign_config_verification_profile(self) -> None:
        cfg = CampaignConfig("verification")
        prof = cfg.get_profile()
        self.assertEqual(prof["max_steps"], 100)
        self.assertEqual(prof["precision"], "fp32")
        self.assertIn("iveri", prof["models"])

    def test_campaign_config_pilot_profile(self) -> None:
        cfg = CampaignConfig("pilot")
        prof = cfg.get_profile()
        self.assertEqual(prof["max_steps"], 5000)
        self.assertEqual(len(prof["seeds"]), 2)

    def test_campaign_config_paper_profile(self) -> None:
        cfg = CampaignConfig("paper")
        prof = cfg.get_profile()
        self.assertEqual(prof["precision"], "bf16")
        self.assertEqual(len(prof["seeds"]), 5)

    def test_campaign_config_invalid_fallback(self) -> None:
        cfg = CampaignConfig("invalid_profile_name")
        self.assertEqual(cfg.profile_name, "pilot")

    # --- 5-9. Execution Backend Configuration Tests ---
    def test_execution_backend_local(self) -> None:
        be = ExecutionBackend("local")
        overrides = be.get_hardware_overrides()
        self.assertEqual(overrides["num_workers"], 0)
        self.assertFalse(overrides["pin_memory"])

    def test_execution_backend_rtx3050(self) -> None:
        be = ExecutionBackend("rtx3050")
        overrides = be.get_hardware_overrides()
        self.assertEqual(overrides["gradient_accumulation"], 16)
        self.assertEqual(overrides["precision"], "fp16")

    def test_execution_backend_kaggle(self) -> None:
        be = ExecutionBackend("kaggle")
        overrides = be.get_hardware_overrides()
        self.assertTrue(overrides["pin_memory"])
        self.assertEqual(overrides["precision"], "bf16")

    def test_execution_backend_lambda(self) -> None:
        be = ExecutionBackend("lambda")
        overrides = be.get_hardware_overrides()
        self.assertEqual(overrides["num_workers"], 8)
        self.assertEqual(overrides["gradient_accumulation"], 2)

    def test_execution_backend_fallback(self) -> None:
        be = ExecutionBackend("nividia_h100_supercomputer")
        self.assertEqual(be.backend_name, "local")

    # --- 10-13. Cost Estimator Tests ---
    def test_cost_estimator_local(self) -> None:
        est = CostEstimator("local")
        res = est.estimate_costs(num_steps=1000, num_seeds=1, num_models=1)
        self.assertEqual(res["estimated_cloud_cost_usd"], 0.0)
        self.assertGreater(res["estimated_gpu_hours"], 0.0)

    def test_cost_estimator_vast_cloud(self) -> None:
        est = CostEstimator("vast")
        res = est.estimate_costs(num_steps=10000, num_seeds=5, num_models=4)
        self.assertGreater(res["estimated_cloud_cost_usd"], 0.0)
        self.assertGreater(res["estimated_energy_kwh"], 0.0)

    def test_cost_estimator_checkpoints_storage(self) -> None:
        est = CostEstimator("local")
        res = est.estimate_costs(num_steps=5000, num_seeds=2, num_models=2, checkpoint_interval=1000)
        self.assertEqual(res["estimated_checkpoints_saved"], 20)  # 5 * 2 * 2
        self.assertAlmostEqual(res["estimated_disk_space_gb"], 0.8)  # 20 * 0.04

    def test_cost_estimator_report_counts(self) -> None:
        # PublicationManager generates exactly 17 Markdown reports
        est = CostEstimator("lambda")
        res = est.estimate_costs(100, 1, 1)
        self.assertEqual(res["estimated_report_count"], 17)

    # --- 14-17. Campaign Dataset Validator Tests ---
    def test_dataset_validator_missing_dir(self) -> None:
        val = CampaignDatasetValidator(data_dir=str(Path(self.tmp_dir) / "missing_data"))
        res = val.validate_processed_datasets()
        self.assertFalse(res["ok"])
        self.assertTrue(any("does not exist" in e for e in res["errors"]))

    def test_dataset_validator_missing_binaries(self) -> None:
        data_path = Path(self.tmp_dir) / "data"
        data_path.mkdir()
        val = CampaignDatasetValidator(data_dir=str(data_path))
        res = val.validate_processed_datasets()
        self.assertFalse(res["ok"])
        self.assertTrue(any("Missing processed data binary" in e for e in res["errors"]))

    def test_dataset_validator_success_and_manifest(self) -> None:
        data_path = Path(self.tmp_dir) / "data"
        data_path.mkdir()
        # Create mock binaries
        with open(data_path / "pretrain.bin", "wb") as f:
            f.write(b"Hello world train dataset")
        with open(data_path / "validation.bin", "wb") as f:
            f.write(b"Hello world val dataset")
        # Create VERSION.json
        meta = {"license": "Apache-2.0"}
        with open(data_path / "VERSION.json", "w") as f:
            json.dump(meta, f)

        val = CampaignDatasetValidator(data_dir=str(data_path))
        res = val.validate_processed_datasets()
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["errors"]), 0)
        self.assertTrue(Path(res["manifest_path"]).exists())

    def test_dataset_validator_manifest_content(self) -> None:
        data_path = Path(self.tmp_dir) / "data"
        data_path.mkdir()
        with open(data_path / "pretrain.bin", "wb") as f:
            f.write(b"Hello world train dataset")
        with open(data_path / "validation.bin", "wb") as f:
            f.write(b"Hello world val dataset")
        with open(data_path / "VERSION.json", "w") as f:
            json.dump({"license": "GPL-3.0"}, f)

        val = CampaignDatasetValidator(data_dir=str(data_path))
        val.validate_processed_datasets()

        manifest_file = data_path / "dataset_manifest.json"
        with open(manifest_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["license"], "GPL-3.0")
        self.assertEqual(data["dataset_version"], "v1.0.0-production")

    # --- 18-21. Checkpoint Lifecycle & Golden Manager Tests ---
    def test_golden_checkpoint_stage_column_ensured(self) -> None:
        # Initializing the GoldenCheckpointManager checks and ensures stage column exists
        manager = GoldenCheckpointManager(self.config, registry=self.registry)
        conn = self.registry._get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(checkpoints)")
            cols = [col[1] for col in cursor.fetchall()]
            self.assertIn("stage", cols)
        finally:
            conn.close()

    def test_golden_checkpoint_promotion(self) -> None:
        manager = GoldenCheckpointManager(self.config, registry=self.registry)
        self.registry.register_experiment("exp1", "purpose1", "H1", "hash1", "git_sha", "main", 42, [])
        self.registry.register_checkpoint("ckpt1", "exp1", 1000, "/path/ckpt1", "hash_ckpt", 10000000)

        manager.promote_checkpoint("ckpt1", "Validated")
        conn = self.registry._get_connection()
        try:
            row = conn.execute("SELECT stage FROM checkpoints WHERE checkpoint_id = 'ckpt1'").fetchone()
            self.assertEqual(row[0], "Validated")
        finally:
            conn.close()

    def test_golden_checkpoint_invalid_stage_rejected(self) -> None:
        manager = GoldenCheckpointManager(self.config, registry=self.registry)
        with self.assertRaises(ValueError):
            manager.promote_checkpoint("ckpt1", "INVALID_LIFECYCLE_STAGE")

    def test_golden_checkpoint_top_screening(self) -> None:
        manager = GoldenCheckpointManager(self.config, registry=self.registry)
        self.registry.register_experiment("exp_screen", "purpose", "H1", "hash1", "git_sha", "main", 42, [])
        self.registry.log_metrics("exp_screen", step=100, train_loss=1.2, val_loss=1.1, perplexity=3.5, accuracy=0.6)
        self.registry.log_metrics("exp_screen", step=200, train_loss=1.0, val_loss=0.9, perplexity=3.1, accuracy=0.7)
        self.registry.log_metrics("exp_screen", step=300, train_loss=0.8, val_loss=0.7, perplexity=2.8, accuracy=0.8)

        tops = manager.select_top_checkpoints("exp_screen")
        self.assertEqual(tops["val_loss"], [300, 200, 100])
        self.assertEqual(tops["perplexity"], [300, 200, 100])
        self.assertEqual(tops["accuracy"], [300, 200, 100])

    # --- 22-25. Campaign Lock Tests ---
    def test_campaign_lock_cycle(self) -> None:
        lock_file = str(Path(self.tmp_dir) / "lock.json")
        lock = CampaignLock(lock_file)
        self.assertFalse(lock.is_locked())

        lock.acquire_lock("hash1", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})
        self.assertTrue(lock.is_locked())

        compliant, violations = lock.verify_lock_compliance("hash1", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})
        self.assertTrue(compliant)
        self.assertEqual(len(violations), 0)

        lock.release_lock()
        self.assertFalse(lock.is_locked())

    def test_campaign_lock_compliance_violation(self) -> None:
        lock_file = str(Path(self.tmp_dir) / "lock.json")
        lock = CampaignLock(lock_file)
        lock.acquire_lock("hash1", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})

        # Violate git SHA
        compliant, violations = lock.verify_lock_compliance("hash1", "git_sha_modified", {"data": "h_data"}, {"ckpt": "h_ckpt"})
        self.assertFalse(compliant)
        self.assertTrue(any("Codebase version mismatch" in v for v in violations))

    def test_campaign_lock_compliance_config_violation(self) -> None:
        lock_file = str(Path(self.tmp_dir) / "lock.json")
        lock = CampaignLock(lock_file)
        lock.acquire_lock("hash1", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})

        # Violate config hash
        compliant, violations = lock.verify_lock_compliance("hash_modified", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})
        self.assertFalse(compliant)
        self.assertTrue(any("Configuration mismatch" in v for v in violations))

    def test_campaign_lock_compliance_datasets_violation(self) -> None:
        lock_file = str(Path(self.tmp_dir) / "lock.json")
        lock = CampaignLock(lock_file)
        lock.acquire_lock("hash1", "git_sha", {"data": "h_data"}, {"ckpt": "h_ckpt"})

        # Violate dataset hashes
        compliant, violations = lock.verify_lock_compliance("hash1", "git_sha", {"data": "h_data_modified"}, {"ckpt": "h_ckpt"})
        self.assertFalse(compliant)
        self.assertTrue(any("Dataset 'data' has changed" in v for v in violations))

    # --- 26-29. Campaign Health Monitor Tests ---
    def test_health_monitor_ok(self) -> None:
        mon = CampaignHealthMonitor()
        status, msg = mon.check_health({"loss": 1.2}, {"gpu_utilization": 85.0})
        self.assertEqual(status, "OK")

    def test_health_monitor_stalled(self) -> None:
        mon = CampaignHealthMonitor()
        # Feed 5 identical loss steps
        for _ in range(6):
            status, msg = mon.check_health({"loss": 1.2}, {"gpu_utilization": 85.0})
        self.assertEqual(status, "PAUSED")
        self.assertIn("stalled training", msg)

    def test_health_monitor_nan_loss(self) -> None:
        mon = CampaignHealthMonitor()
        status, msg = mon.check_health({"loss": float("nan")}, {"gpu_utilization": 85.0})
        self.assertEqual(status, "PAUSED")
        self.assertIn("loss is nan", msg.lower())

    def test_health_monitor_cost_limit(self) -> None:
        mon = CampaignHealthMonitor(cost_limit_usd=10.0)
        status, msg = mon.check_health({"loss": 1.1}, {"gpu_utilization": 85.0, "accumulated_cost_usd": 15.0})
        self.assertEqual(status, "PAUSED")
        self.assertIn("exceeded budget", msg)

    # --- 30-31. Experiment Manifest Generator Tests ---
    def test_manifest_generator_output(self) -> None:
        gen = ExperimentManifestGenerator()
        out_path = Path(self.tmp_dir) / "experiment_manifest.json"
        res = gen.generate_manifest(
            experiment_id="exp_test",
            config_hash="conf_hash",
            dataset_hashes={"train": "h1"},
            checkpoint_hashes={"golden": "h2"},
            output_path=out_path,
        )
        self.assertTrue(out_path.exists())
        self.assertEqual(res["config_hash"], "conf_hash")
        self.assertEqual(res["versions"]["pytorch"], torch.__version__)

    def test_manifest_generator_system_details(self) -> None:
        gen = ExperimentManifestGenerator()
        out_path = Path(self.tmp_dir) / "experiment_manifest.json"
        env_lock_path = Path(self.tmp_dir) / "environment.txt"
        res = gen.generate_manifest("exp_test", "conf_hash", {}, {}, out_path)
        gen.generate_environment_lock(env_lock_path)
        self.assertIn("os", res["environment"])
        self.assertIn("hostname", res["environment"])
        self.assertTrue(env_lock_path.exists())

    # --- 32-33. Provider-Agnostic External Grader Tests ---
    def test_external_evaluator_skips_on_missing_key(self) -> None:
        evaluator = ExternalModelEvaluator()
        # Ensure env key is deleted to trigger clean skip
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        res = evaluator.evaluate_model("openai", ["test prompt"])
        self.assertEqual(res["status"], "Not Evaluated")
        self.assertIsNone(res["correctness_score"])
        self.assertIn("Missing credentials variable", res["reason"])

    def test_external_evaluator_evaluates_on_present_key(self) -> None:
        evaluator = ExternalModelEvaluator()
        os.environ["OPENAI_API_KEY"] = "mock_api_key_12345"
        try:
            res = evaluator.evaluate_model("openai", ["test prompt"])
            self.assertEqual(res["status"], "Evaluated")
            self.assertIsNotNone(res["correctness_score"])
        finally:
            del os.environ["OPENAI_API_KEY"]

    # --- 34-35. Publication Manager Verification Tests ---
    def test_publication_manager_compilation(self) -> None:
        experiment_id = seed_measured_experiment(self.registry, "exp_run_1", with_benchmark=True)
        pub = PublicationManager(self.registry, output_dir=self.tmp_dir)
        git_sha = GIT_SHA
        config_hash = CONFIG_HASH
        dataset_hashes = {"data": "h_data"}
        checkpoint_hashes = {"ckpt": "h_ckpt"}

        manifest = pub.generate_and_verify_all_assets(
            experiment_id=experiment_id,
            git_sha=git_sha,
            config_hash=config_hash,
            dataset_hashes=dataset_hashes,
            checkpoint_hashes=checkpoint_hashes,
            random_seed=42,
        )

        self.assertIn("Figure_1_loss_curves", manifest["assets"])
        self.assertIn("Table_1_benchmarks", manifest["assets"])

        # Check files existence
        self.assertTrue((Path(self.tmp_dir) / "publication" / "paper_manifest.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "reproducibility_package.zip").exists())

    def test_campaign_runner_dry_run_flag(self) -> None:
        from research.campaign_runner import CampaignRunner
        # Create validation directory structure
        data_path = Path(self.tmp_dir) / "data"
        data_path.mkdir()
        with open(data_path / "pretrain.bin", "wb") as f:
            f.write(b"Hello world")
        with open(data_path / "validation.bin", "wb") as f:
            f.write(b"Hello world")
        with open(data_path / "VERSION.json", "w") as f:
            json.dump({"license": "MIT"}, f)

        # Instantiate runner with dry-run=True (Phase 5 constructor includes new kwargs)
        runner = CampaignRunner(
            profile_name="pilot",
            backend_name="local",
            db_path=self.db_path,
            output_dir=self.tmp_dir,
            resume_strategy="FROM_BEST",
            ablation="none",
            stage="pretrain",
            benchmarks_only=False,
            dry_run=True,
        )
        runner.data_val = CampaignDatasetValidator(data_dir=str(data_path))

        res = runner.run_campaign()
        self.assertEqual(res["status"], "DRY_RUN_COMPLETED")
        self.assertEqual(runner.resume_strategy, "FROM_BEST")
        self.assertTrue(runner.dry_run)

        def _mock_pretraining(
            runner_self: CampaignRunner,
            exp_id: str,
            model_name: str,
            seed: int,
            profile: dict,
            ablation_overrides: dict,
        ) -> dict:
            if not getattr(_mock_pretraining, "_benchmark_ready", False):
                runner_self.registry.register_benchmark(
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
                _mock_pretraining._benchmark_ready = True
            run_id = f"bench_{exp_id}"
            runner_self.registry.log_benchmark_run(
                run_id=run_id,
                experiment_id=exp_id,
                benchmark_id="HumanEval",
                step=100,
                score=0.85,
                provenance_label="MEASURED",
            )
            runner_self.registry.log_benchmark_integrity(
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
            return {
                "final_loss": 1.5,
                "final_val_loss": 1.4,
                "final_perplexity": 4.0,
                "checkpoint_dir": str(runner_self.output_dir),
            }

        mock_env = {
            "os": "test",
            "python_version": "3.14",
            "pytorch_version": torch.__version__,
            "numpy_version": "2.0",
            "gpu": "CPU",
            "cuda_driver": "none",
            "git_sha": GIT_SHA,
            "git_branch": "main",
            "pip_freeze": "",
        }

        # Run non-dry-run verification campaign with measured training mocked
        runner_real = CampaignRunner(
            profile_name="verification",
            backend_name="local",
            db_path=self.db_path,
            output_dir=self.tmp_dir,
            resume_strategy="AUTO",
            ablation="none",
            stage="pretrain",
            benchmarks_only=False,
            dry_run=False,
        )
        runner_real.data_val = CampaignDatasetValidator(data_dir=str(data_path))
        with (
            patch.object(CampaignRunner, "_attempt_real_pretraining", _mock_pretraining),
            patch.object(BenchmarkIntegrityFramework, "get_env_info", return_value=mock_env),
        ):
            res_real = runner_real.run_campaign()
        self.assertEqual(res_real["status"], "SUCCESS")

        self.assertTrue((Path(self.tmp_dir) / "experiment_manifest.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "environment.txt").exists())
        self.assertTrue((Path(self.tmp_dir) / "publication" / "Evidence_Index.md").exists())
        self.assertTrue((Path(self.tmp_dir) / "campaign_manifest.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "publication_manifest.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "evidence_graph.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "artifact_dag.json").exists())
        self.assertTrue((Path(self.tmp_dir) / "Campaign_Certificate.md").exists())
        self.assertTrue((Path(self.tmp_dir) / "Phase_6_3_Freeze.md").exists())
        self.assertTrue((Path(self.tmp_dir) / "FINAL_REPORT.md").exists())

        # Replay should fail closed when H1-H10 claim chain is incomplete
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, "replay_campaign.py", "--db-path", self.db_path, "--output-dir", self.tmp_dir, "--reviewer-mode"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("BROKEN", res.stdout)


    # ─── Phase 5 Specific Tests ───────────────────────────────────────────────

    def test_ablation_config_overrides_recognized(self) -> None:
        """Verify ABLATION_CONFIG_OVERRIDES maps all expected ablation keys."""
        from research.campaign_runner import ABLATION_CONFIG_OVERRIDES
        expected_keys = {"none", "no_titans", "no_blt", "no_mor", "no_moe", "no_entropy_routing"}
        self.assertEqual(set(ABLATION_CONFIG_OVERRIDES.keys()), expected_keys)
        self.assertEqual(ABLATION_CONFIG_OVERRIDES["none"], {})
        self.assertIn("use_titans", ABLATION_CONFIG_OVERRIDES["no_titans"])
        self.assertFalse(ABLATION_CONFIG_OVERRIDES["no_titans"]["use_titans"])

    def test_ablation_overrides_apply_to_model_config(self) -> None:
        from configs.base_config import apply_ablation_overrides, get_base_config
        from research.campaign_runner import ABLATION_CONFIG_OVERRIDES

        cfg = get_base_config()
        apply_ablation_overrides(cfg, ABLATION_CONFIG_OVERRIDES["no_moe"])
        self.assertFalse(cfg.model.use_moe)

    def test_campaign_runner_ablation_stage_accepted(self) -> None:
        """Verify CampaignRunner accepts ablation and stage kwargs without error."""
        runner = CampaignRunner(
            profile_name="verification",
            backend_name="local",
            db_path=self.db_path,
            output_dir=self.tmp_dir,
            ablation="no_titans",
            stage="pretrain",
            benchmarks_only=False,
            dry_run=True,
        )
        self.assertEqual(runner.ablation, "no_titans")
        self.assertEqual(runner.stage, "pretrain")
        self.assertFalse(runner.benchmarks_only)

    def test_campaign_runner_benchmarks_only_accepted(self) -> None:
        """Verify benchmarks_only flag is stored correctly."""
        runner = CampaignRunner(
            profile_name="verification",
            backend_name="local",
            db_path=self.db_path,
            output_dir=self.tmp_dir,
            benchmarks_only=True,
            dry_run=True,
        )
        self.assertTrue(runner.benchmarks_only)

    def test_final_report_generation(self) -> None:
        """Verify generate_final_report() writes FINAL_REPORT.md with all 17 report entries."""
        seed_measured_experiment(self.registry, "exp_final_report", with_benchmark=True)
        pub = PublicationManager(self.registry, output_dir=self.tmp_dir)
        pub.generate_final_report(campaign_id="TEST_CAMPAIGN_PHASE5")
        final_report = Path(self.tmp_dir) / "FINAL_REPORT.md"
        self.assertTrue(final_report.exists())
        content = final_report.read_text(encoding="utf-8")
        # Must reference all 17 reports
        expected_reports = [
            "Training_Report.md", "Baseline_Report.md", "Ablation_Report.md",
            "Instruction_Report.md", "Coding_Report.md", "Alignment_Report.md",
            "Calibration_Report.md", "Efficiency_Report.md", "Energy_Report.md",
            "Long_Context_Report.md", "Statistics_Report.md", "Hypothesis_Report.md",
            "Architecture_Validation_Report.md", "Reproducibility_Report.md",
            "Campaign_Report.md", "Evidence_Index.md", "Executive_Summary.md",
        ]
        for report_name in expected_reports:
            self.assertIn(report_name, content, f"FINAL_REPORT.md missing link to {report_name}")
        # Must reference campaign ID
        self.assertIn("TEST_CAMPAIGN_PHASE5", content)
        # Must have replay instructions
        self.assertIn("replay_campaign.py", content)

    def test_campaign_paper_profile_has_checkpoint_interval(self) -> None:
        """Regression: paper profile must define checkpoint_interval (was previously missing)."""
        cfg = CampaignConfig("paper")
        prof = cfg.get_profile()
        self.assertIn("checkpoint_interval", prof)
        self.assertIsInstance(prof["checkpoint_interval"], int)
        self.assertGreater(prof["checkpoint_interval"], 0)


if __name__ == "__main__":
    unittest.main()
