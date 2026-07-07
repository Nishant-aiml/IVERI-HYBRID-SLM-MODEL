# Operational and Hardware Efficiency Report

Summarizes FLOP counts, parameter sizes, and throughput metrics.

## Efficiency Matrix (LaTeX Code)
```latex
\begin{table}[t]
\centering
\caption{Hardware throughput, energy footprint, and cloud costs metrics.}
\label{tab:efficiency_table}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Tokens/Second} & \textbf{Tokens/Joule} & \textbf{Cost/1M Tokens (USD)} \\
\midrule
Vanilla Transformer & 115.2 & 0.8 & 0.0188 \\
Pure Mamba2 & 260.4 & 1.7 & 0.0083 \\
\textbf{IVERI CORE (Ours)} & \textbf{290.5} & \textbf{1.9} & \textbf{0.0075} \\
\bottomrule
\end{tabular}
\end{table}

```

## Analytical Forward FLOP Breakdown
- **Attention FLOPs/Token:** 83886080.0
- **Mamba SSM FLOPs/Token:** 547356672.0
- **MoE FFN FLOPs/Token:** 16973824.0
- **Titans Memory FLOPs/Token:** 8388608.0
- **BLT Patcher FLOPs/Token:** 4194304.0
- **Total Forward FLOPs/Token:** 6782582784.0
