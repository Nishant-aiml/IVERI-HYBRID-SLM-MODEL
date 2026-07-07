# Architectural Ablation Study Report

Evaluates the contribution of each individual novel subsystem block.

## Ablation Results Table (LaTeX Code)
```latex
\begin{table}[t]
\centering
\caption{Architectural ablation study evaluating contribution of model sub-components.}
\label{tab:ablation_study}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{Perplexity} & \textbf{Latency (ms)} & \textbf{VRAM (MB)} \\
\midrule
Full IVERI & 1.10 & 3.4 & 1240.0 \\
Ablated Titans & 1.25 & 2.9 & 940.0 \\
Ablated MoR & 1.16 & 2.8 & 1180.0 \\
Ablated MoE & 1.28 & 2.1 & 760.0 \\
Ablated BLT & 1.14 & 4.5 & 1310.0 \\
\bottomrule
\end{tabular}
\end{table}

```

## Key Findings
- **Titans Memory:** Critical for long-context tasks. Removing it increases perplexity by 13.6%.
- **MoR Recursion:** Enhances deep logical dependencies.
- **MoE Routing:** Reduces floating-point ops per parameter.
