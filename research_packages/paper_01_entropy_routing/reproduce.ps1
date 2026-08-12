$ErrorActionPreference = "Stop"
Write-Host "==============================================================="
Write-Host "=== REPRODUCING PAPER: paper_01_entropy_routing ==="
Write-Host "==============================================================="
python proof/verify_everything.py
python proof/verify_figures.py
python proof/verify_tables.py
python proof/scientific_ci.py
Write-Host "[+] Reproduction Complete!"
