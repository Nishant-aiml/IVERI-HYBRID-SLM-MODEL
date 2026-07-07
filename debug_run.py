import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["IVERI_DISABLE_HF"] = "1"
os.environ["WANDB_MODE"] = "offline"

import sys
import traceback
from pathlib import Path
import logging

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add workspace root to python path to avoid shadowing
workspace = "C:/Users/datta.000/Desktop/iveri core nexus/iveri-core"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

from research.campaign_runner import CampaignRunner
from research.campaign_config import CampaignConfig
from research.campaign_dataset_validator import CampaignDatasetValidator
from research.benchmark_integrity import BenchmarkIntegrityFramework
from research.campaign_lock import CampaignLock
from research.experiment_registry import ExperimentRegistry

def main():
    print("Starting step-by-step debug runner...")
    try:
        # 1. Config profile
        print("Step 1: Resolving config profile...")
        config_mgr = CampaignConfig("paper")
        profile = config_mgr.get_profile()
        print("Profile resolved:", profile)

        # 2. Dataset validation
        print("Step 2: Running dataset validator...")
        data_val = CampaignDatasetValidator()
        data_checks = data_val.validate_processed_datasets()
        print("Dataset validation result:", data_checks)

        # 3. Benchmark integrity
        print("Step 3: Running benchmark integrity validation...")
        integrity_fw = BenchmarkIntegrityFramework(db_path="research/experiments.db")
        dataset_locks = integrity_fw.lock_dataset_revisions()
        env_info = integrity_fw.get_env_info()
        print("Benchmark integrity locks resolved. Env info:", env_info)

        # 4. Campaign lock
        print("Step 4: Managing campaign lock...")
        lock_mgr = CampaignLock()
        config_hash = "fe12de98d498ce0be38e504594f17bf91a78689000e67cd706d4b0a8e5da8f18" # mock/rep config hash
        git_sha = env_info.get("git_sha", "unknown")
        dataset_hashes = {Path(k).name: v for k, v in dataset_locks.items()}
        checkpoint_hashes = {"golden_base": "hash_golden_phase6_1"}
        
        lock_mgr.acquire_lock(config_hash, git_sha, dataset_hashes, checkpoint_hashes)
        compliant, lock_violations = lock_mgr.verify_lock_compliance(
            config_hash, git_sha, dataset_hashes, checkpoint_hashes
        )
        print("Lock compliance check result: compliant =", compliant, "violations =", lock_violations)

        # 5. Registry instantiation
        print("Step 5: Instantiating registry...")
        registry = ExperimentRegistry("research/experiments.db")
        print("Registry instantiated successfully.")

        # 6. Dispatching first experiment registration
        print("Step 6: Registering first experiment in DB...")
        ablation_tag = "pretrain"
        model_name = "iveri"
        seed = 42
        exp_id = f"IVERI_Phase5_{ablation_tag}_Seed{seed}_{model_name.upper()}_Run001"
        
        registry.register_experiment(
            experiment_id=exp_id,
            purpose="Debug registration",
            hypothesis="H1",
            config_hash=config_hash,
            git_sha=git_sha,
            git_branch="main",
            random_seed=seed,
            tags=[model_name, "phase5", "paper", ablation_tag],
        )
        print("First experiment registered successfully in DB!")

        # 7. Attempt real pretraining
        print("Step 7: Dispatching pretraining runner...")
        from training.pretrain_runner import run_pretraining
        from configs.base_config import get_base_config
        cfg = get_base_config()
        
        # Override dangerous settings on Windows
        cfg.hardware.num_workers = 0
        print("Base config loaded:", cfg)
        
        print("Calling run_pretraining...")
        res = run_pretraining(
            config=cfg,
            verification_level=3,
            run_baseline=False,
            dataset_name="tinystories",
        )
        print("run_pretraining finished with result:", res)
        
    except BaseException as e:
        print("Exception caught in script:", e)
        traceback.print_exc()
        sys.exit(1)
    print("Step-by-step debug runner finished successfully!")

if __name__ == '__main__':
    main()
