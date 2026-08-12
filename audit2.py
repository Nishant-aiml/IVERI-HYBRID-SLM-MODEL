import os

base_dir = r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core"
report = []

def analyze_file(filepath):
    try:
        size = os.path.getsize(filepath)
        if size == 0: return "EMPTY", size
        
        # Determine real/placeholder for text
        if filepath.endswith(('.md', '.json', '.csv', '.jsonl', '.txt', '.log')):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(500).lower()
                
            if 'lorem ipsum' in content or 'placeholder' in content or 'TODO' in content or 'mock' in content or 'template' in content:
                # also check if the file size is very small for an image/figure
                return "TEMPLATE/PLACEHOLDER/MOCK", size
            
            # Simple heuristic
            return "REAL DATA", size
            
        if filepath.endswith(('.png', '.svg')):
            if size < 100:
                return "MOCK IMAGE (Too small)", size
            return "REAL IMAGE", size
            
        if filepath.endswith('.pt'):
            if size == 0:
                return "EMPTY CHECKPOINT", size
            return "REAL CHECKPOINT", size
            
        return "UNKNOWN", size
        
    except Exception as e:
        return f"ERROR: {str(e)}", 0

for d in ["reports", "logs", "paper", "research", "research_log", "experiments", "archives", "docs", "data/certified", "data/curriculum", "data/raw"]:
    path = os.path.join(base_dir, d)
    if not os.path.exists(path):
        report.append(f"Directory {d}/ is MISSING.")
        continue
        
    report.append(f"\n--- Directory {d}/ ---")
    file_count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_count += 1
            if file_count > 100 and d == "reports":
                # Sample if too many
                continue
                
            fp = os.path.join(root, file)
            rel_path = os.path.relpath(fp, base_dir)
            status, size = analyze_file(fp)
            report.append(f"{rel_path}: {status} ({size} bytes)")
            
    if file_count == 0:
        report.append("EMPTY DIRECTORY")

with open(r"c:\Users\datta.000\Desktop\iveri core nexus\iveri-core\final_audit_summary.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(report))
