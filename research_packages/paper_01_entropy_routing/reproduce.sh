#!/usr/bin/env bash
set -e
echo "==============================================================="
echo "=== REPRODUCING PAPER: paper_01_entropy_routing ==="
echo "==============================================================="
python proof/verify_everything.py
python proof/verify_figures.py
python proof/verify_tables.py
python proof/scientific_ci.py
echo "[+] Reproduction Complete!"
