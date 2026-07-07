# Dataset Statistics Reporting Specification

This report details how size stats, length histograms, and composition graphs are generated.

## 1. Metric Calculations

All metrics are computed on the byte level (B, KB, MB, GB, TB) rather than subword tokens to match the byte-level processing nature of IVERI:

- **average_bytes**: Mean length of byte representations.
- **median_bytes**: Median (50th percentile).
- **percentiles**: p25, p75, p95, p99 to inspect dataset length skew.
- **empty_doc_count**: Number of documents containing 0 bytes.

## 2. Length Histograms

Histograms are computed using numpy's `histogram` function with a configurable number of bins. Bin boundaries and counts are saved in both Markdown tables and JSON summaries.

## 3. Pie Chart Composition (JSON)

The `generate_composition_json` method produces a JSON structure containing:
- Dataset name.
- Byte count.
- Percentage weight in the overall stage mixture.
- License.
- Stage.

This structure allows dynamic rendering of dataset composition pie charts in dashboards or experiment trackers.
