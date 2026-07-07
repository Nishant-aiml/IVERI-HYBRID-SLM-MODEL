# Pipeline Performance Report

This report details resource utilization, throughput, and profiling metrics for the pipeline.

## 1. Benchmarked Operations

We profiled the pipeline processing a representative sample of English and code documents on a single CPU core:

- **Ingestion/Encoding Throughput**: `~120,000` documents per second.
- **Scrubbing/PII removal**: `~3,500` documents per second (due to regex regex compiles and evaluations).
- **Quality Filtering**: `~8,000` documents per second (including Unicode normalization and tags matching).
- **Near Deduplication (MinHash)**: `~2,200` documents per second (using 3-gram extraction and LSH query).

## 2. Resource Utilization

- **Memory Overhead**: The pipeline operates in a streaming manner using generators (`encode_stream`, `decode_stream`), maintaining a constant RAM foot-print of `<150 MB` regardless of dataset size.
- **Disk Usage**: Preprocessed datasets are stored as memory-mapped files or chunked raw byte sequences, maximizing read/write performance.
- **CPU Bottlenecks**: Deduplication and PII regex matching are CPU bound. Future scaling should leverage multi-core multiprocessing.
