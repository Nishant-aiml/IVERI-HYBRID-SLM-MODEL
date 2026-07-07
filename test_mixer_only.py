import sys
workspace = "C:/Users/datta.000/Desktop/iveri core nexus/iveri-core"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

print("Importing MixingStrategy...")
from data.pipeline.mixer import MixingStrategy
print("MixingStrategy imported successfully!")
