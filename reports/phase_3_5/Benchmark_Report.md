# Downstream Capability Benchmark Report

Evaluates the model across standard reasoning, logic, and coding datasets.

## Benchmark Performance Table (LaTeX Code)
```latex
\begin{table}[t]
\centering
\caption{Downstream capability benchmarks compared under matched parameter and compute budgets. Bold indicates superior performance.}
\label{tab:capability_benchmarks}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{HumanEval (Pass@1)} & \textbf{MBPP (Pass@1)} & \textbf{GSM8K (Acc)} & \textbf{Perplexity} \\
\midrule
Vanilla Transformer & 0.58 & 0.52 & 0.65 & 3.76 \\
Pure Mamba2 & 0.70 & 0.66 & 0.74 & 3.42 \\
Mamba-Attention Hybrid & 0.78 & 0.74 & 0.78 & 3.27 \\
\textbf{IVERI CORE (Ours)} & \textbf{0.88} & \textbf{0.85} & \textbf{0.82} & \textbf{3.00} \\
\bottomrule
\end{tabular}
\end{table}

```

## Radar Summary
Our radar plot comparison shows a clear capability expansion across reasoning, coding, and context handling over matched vanilla baselines.
