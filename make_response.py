import os
import re

base_dir = r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core"
dirs_to_check = [
    "reports", "logs", "paper", "research", "research_log", "experiments", "archives", "docs",
    "data/certified", "data/curriculum", "data/raw"
]

output = ["# IVERI Core Repository Audit\n"]

def get_status(content, size, filepath):
    if size == 0: return "Empty"
    if filepath.endswith(('.png', '.svg')):
        if size == 23: return "Placeholder/Empty Image (23 bytes)"
        return f"Image Data ({size} bytes)"
    if filepath.endswith('.pt'):
        return f"Checkpoint Data ({size} bytes - looks small for a real large model)" if size < 10000000 else f"Real Checkpoint ({size} bytes)"
    
    content = content.lower()
    if 'lorem ipsum' in content or 'placeholder' in content: return "Mock/Placeholder Text"
    if 'mock' in content and filepath.endswith('.json'): return "Mock JSON"
    if 'template' in content: return "Template"
    if filepath.endswith('.md'):
        if "verdict" in content or "audit" in content or "status" in content:
            return "Generated Report/Audit Text"
        return "Documentation/Text"
    if filepath.endswith('.json'): return "JSON Data"
    if filepath.endswith('.jsonl'): return "JSONL Data"
    if filepath.endswith('.csv'): return "CSV Data"
    if filepath.endswith('.py'): return "Python Source"
    
    return "Data"

for d in dirs_to_check:
    path = os.path.join(base_dir, d)
    if not os.path.exists(path):
        output.append(f"## {d}/\nStatus: Directory Missing\n")
        continue
    
    output.append(f"## {d}/")
    files_found = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            files_found += 1
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, base_dir)
            size = os.path.getsize(filepath)
            
            preview = ""
            try:
                if file.endswith(('.md', '.json', '.csv', '.txt', '.log', '.jsonl', '.py')):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        preview = f.read(150)
                else:
                    preview = "[Binary]"
            except:
                pass
                
            status = get_status(preview, size, filepath)
            preview_clean = preview.replace('\n', ' ').strip()[:80] + ('...' if len(preview)>80 else '')
            output.append(f"- **{rel_path}** | {size} B | Type: {status} | Evidence: `{preview_clean}`")
            
    if files_found == 0:
        output.append("Status: Empty Directory\n")
    output.append("")

output.append("## Checkpoints (.pt)")
for root, dirs, files in os.walk(base_dir):
    if ".venv" in root or "__pycache__" in root: continue
    for file in files:
        if file.endswith('.pt'):
            fp = os.path.join(root, file)
            size = os.path.getsize(fp)
            rel_path = os.path.relpath(fp, base_dir)
            output.append(f"- {rel_path}: {size} bytes")

output.append("\n## Figures (.png, .svg)")
for root, dirs, files in os.walk(base_dir):
    if ".venv" in root or "__pycache__" in root: continue
    for file in files:
        if file.endswith(('.png', '.svg')):
            fp = os.path.join(root, file)
            size = os.path.getsize(fp)
            rel_path = os.path.relpath(fp, base_dir)
            output.append(f"- {rel_path}: {size} bytes")

with open(r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\final_response.md", "w", encoding='utf-8') as f:
    f.write("\n".join(output))
