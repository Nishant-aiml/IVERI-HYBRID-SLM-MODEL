import sys
workspace = "C:/Users/datta.000/Desktop/iveri core nexus/iveri-core"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

print("1. import torch")
import torch

print("2. from configs.base_config import IVERIConfig")
from configs.base_config import IVERIConfig

print("3. from core.constants import ARCHITECTURE_VERSION")
from core.constants import ARCHITECTURE_VERSION

print("4. from evaluation.arch_eval import ArchitectureEvaluator")
from evaluation.arch_eval import ArchitectureEvaluator

print("5. from evaluation.benchmark import InferenceBenchmark")
from evaluation.benchmark import InferenceBenchmark

print("6. from evaluation.checkpoint_compare import CheckpointComparator")
from evaluation.checkpoint_compare import CheckpointComparator

print("7. from evaluation.generation import GenerationEvaluator")
from evaluation.generation import GenerationEvaluator

print("8. from evaluation.memory_tracker import MemoryTracker")
from evaluation.memory_tracker import MemoryTracker

print("9. from evaluation.perplexity import PerplexityEvaluator")
from evaluation.perplexity import PerplexityEvaluator

print("10. from evaluation.report_generator import ReportGenerator")
from evaluation.report_generator import ReportGenerator

print("11. from training.logger import _git_info, _system_info")
from training.logger import _git_info, _system_info

print("12. from training.mixed_precision import PrecisionHandler")
from training.mixed_precision import PrecisionHandler

print("All Evaluator imports succeeded!")
