# 05 ENVIRONMENT REPRODUCIBILITY
- Validated on Windows 11, Python 3.12.10, PyTorch 2.5.1+cu121.
- Hardware limitation: RTX 3050 Laptop with 4.29 GB VRAM heavily constrains context length and batch size.
- VRAM Usage observed during 10-step benchmark:
  - IVERI: 336MB
  - Transformer: 241MB
  - Mamba2: 207MB
