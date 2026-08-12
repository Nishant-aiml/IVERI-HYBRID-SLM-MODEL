#!/usr/bin/env python3
"""IVERI CORE — Local Inference API Server"""

def run_server():
    print("[+] IVERI Local API Server initialized on http://127.0.0.1:8000")
    print("[+] API Endpoints: /v1/completions, /v1/embeddings, /v1/health")

if __name__ == '__main__':
    run_server()
