import os
import json

base_dir = r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core"
dirs_to_check = [
    "reports", "logs", "paper", "research", "research_log", "experiments", "archives", "docs",
    "data/certified", "data/curriculum", "data/raw"
]

output = []

for d in dirs_to_check:
    path = os.path.join(base_dir, d)
    if not os.path.exists(path):
        output.append(f"Directory {d} does not exist.")
        continue
    
    output.append(f"--- Directory: {d} ---")
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, base_dir)
            size = os.path.getsize(filepath)
            
            content_preview = ""
            try:
                if file.endswith(('.md', '.json', '.csv', '.txt', '.log', '.jsonl', '.py')):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content_preview = f.read(500)
                else:
                    content_preview = "[Binary or unsupported format]"
            except Exception as e:
                content_preview = f"[Error reading file: {e}]"
            
            output.append(f"File: {rel_path}")
            output.append(f"Size: {size} bytes")
            output.append(f"Preview:\n{content_preview.strip()[:500]}")
            output.append("-" * 40)

# Also find all .pt, .png, .svg
output.append("--- Global search for specific extensions ---")
for ext in [".pt", ".png", ".svg", ".csv"]:
    output.append(f"Searching for {ext} files...")
    for root, dirs, files in os.walk(base_dir):
        if ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_dir)
                size = os.path.getsize(filepath)
                output.append(f"Found {ext}: {rel_path} - {size} bytes")

with open(r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\audit_report.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(output))

print("Audit script finished.")
