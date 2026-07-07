import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["IVERI_DISABLE_HF"] = "1"  # Disable HF datasets import to prevent deadlocks

import numpy as np  # Import numpy BEFORE torch to prevent OpenMP/MKL initialization deadlock
import torch       # Then import torch

import sys
import traceback

workspace = "C:/Users/datta.000/Desktop/iveri core nexus/iveri-core"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

try:
    print("1. Importing configs.base_config...")
    from configs.base_config import get_base_config
    
    print("2. Importing baselines.baseline_transformer...")
    from baselines.baseline_transformer import BaselineTransformer
    
    print("3. Importing model.iveri_core...")
    from model.iveri_core import IVERIModel
    
    print("4. Importing training.trainer...")
    from training.trainer import Trainer
    
    print("5. Importing training.pretraining_dataset sub-imports...")
    
    print("5.0.1 data.pipeline.byte_counter...")
    import data.pipeline.byte_counter
    
    print("5.0.2 data.pipeline.byte_encoder...")
    import data.pipeline.byte_encoder
    
    print("5.0.3 data.pipeline.data_registry...")
    import data.pipeline.data_registry
    
    print("5.0.4 data.pipeline.dataloader...")
    import data.pipeline.dataloader
    
    print("5.0.5 data.pipeline.deduplication...")
    import data.pipeline.deduplication
    
    print("5.0.6 data.pipeline.downloader...")
    import data.pipeline.downloader
    
    print("5.0.7 data.pipeline.language_detector...")
    import data.pipeline.language_detector
    
    print("5.0.8 data.pipeline.license_checker...")
    import data.pipeline.license_checker
    
    print("5.0.9 data.pipeline.mixer...")
    import data.pipeline.mixer
    
    print("5.0.10 data.pipeline.pii_remover...")
    import data.pipeline.pii_remover
    
    print("5.0.11 data.pipeline.provenance...")
    import data.pipeline.provenance
    
    print("5.0.12 data.pipeline.quality_filter...")
    import data.pipeline.quality_filter
    
    print("5.0.13 data.pipeline.sft_validator...")
    import data.pipeline.sft_validator
    
    print("5.0.14 data.pipeline.splitter...")
    import data.pipeline.splitter
    
    print("5.0.15 data.pipeline.statistics...")
    import data.pipeline.statistics
    
    print("5.0.16 data.pipeline.versioning...")
    import data.pipeline.versioning
    
    print("5.1 configs.base_config...")
    from configs.base_config import IVERIConfig
    print("5.2 data.pipeline.data_registry...")
    from data.pipeline.data_registry import DataRegistry, DatasetEntry
    print("5.3 data.pipeline.dataloader...")
    from data.pipeline.dataloader import PretrainByteDataset
    print("5.4 data.pipeline.license_checker...")
    from data.pipeline.license_checker import LicenseChecker
    print("5.5 data.pipeline.versioning...")
    from data.pipeline.versioning import DatasetVersioner
    print("5.6 importing training.pretraining_dataset itself...")
    from training.pretraining_dataset import PretrainingDatasetLoader
    
    print("6. Importing evaluation.pretraining_eval...")
    from evaluation.pretraining_eval import PretrainingEvaluator
    
    print("7. Importing training.pretrain_runner...")
    from training.pretrain_runner import run_pretraining
    
    print("All imports SUCCESSFUL!")
except BaseException as e:
    print("Error caught in test_imports.py:", e)
    traceback.print_exc()
    sys.exit(1)
