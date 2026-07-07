import sys
workspace = "C:/Users/datta.000/Desktop/iveri core nexus/iveri-core"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

print("Importing CurriculumScheduler...")
from training.curriculum import CurriculumScheduler
print("CurriculumScheduler imported successfully!")
