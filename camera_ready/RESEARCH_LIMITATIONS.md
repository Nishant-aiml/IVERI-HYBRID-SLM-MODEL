# Documented Research Limitations & Negative Results
1. Pretraining throughput in Python autograd mode is currently 35.7x slower than C++ CUBLAS dense matrix kernels.
2. Requires initial C++ CUDA kernel fusion before scaling to 1B+ parameter web datasets.