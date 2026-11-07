@echo off
title IVERI Core HYPERION - Live Multi-Architecture PyTorch Visualizer
cd /d "c:\Users\datta.000\Desktop\iveri core nexus"
echo ===============================================================================
echo     IVERI Core -- HYPERION 10M Live Research Visualizer & Multi-Model Server
echo ===============================================================================
echo Starting live PyTorch foundation model bridge on http://localhost:8088 ...
timeout /t 2 >nul
start "" "http://localhost:8088/"
python -u live_visualizer_server.py 8088
pause
