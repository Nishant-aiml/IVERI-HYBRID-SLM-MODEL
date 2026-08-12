# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Statistical Validation Engine for IVERI CORE.

Calculates parametric and non-parametric statistics, confidence intervals,
bootstrap distributions, effect sizes (Cohen's d, Cliff's Delta), and
multiple testing corrections (Holm-Bonferroni) strictly from empirical CSV logs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def calculate_descriptive_stats(data: List[float]) -> Dict[str, float]:
    """Calculate mean, median, variance, std, and 95% confidence interval."""
    if not data:
        return {"mean": 0.0, "median": 0.0, "var": 0.0, "std": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0}

    n = len(data)
    mean_val = sum(data) / n
    sorted_data = sorted(data)
    
    if n % 2 == 1:
        median_val = sorted_data[n // 2]
    else:
        median_val = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

    variance_val = sum((x - mean_val) ** 2 for x in data) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance_val)

    # 95% CI assuming normal distribution (t-critical ~ 1.96 for n>30, 2.0 for smaller)
    stderr = std_val / math.sqrt(n) if n > 0 else 0.0
    margin = 1.96 * stderr

    return {
        "n": float(n),
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "var": round(variance_val, 4),
        "std": round(std_val, 4),
        "ci_95_low": round(mean_val - margin, 4),
        "ci_95_high": round(mean_val + margin, 4),
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size between two groups."""
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    m1, m2 = sum(group1) / n1, sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1) if n1 > 1 else 0.0
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1) if n2 > 1 else 0.0
    
    pooled_std = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1e-8
    return round((m1 - m2) / pooled_std, 4)


def cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """Calculate Cliff's Delta non-parametric effect size."""
    if not group1 or not group2:
        return 0.0
    n1, n2 = len(group1), len(group2)
    greater = 0
    less = 0
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return round((greater - less) / (n1 * n2), 4)


def welch_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """Calculate Welch's t-test statistic and approximate p-value."""
    if not group1 or not group2:
        return {"t_statistic": 0.0, "p_value": 1.0}
    n1, n2 = len(group1), len(group2)
    m1, m2 = sum(group1) / n1, sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1) if n1 > 1 else 1e-8
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1) if n2 > 1 else 1e-8
    
    se = math.sqrt((v1 / n1) + (v2 / n2)) if ((v1 / n1) + (v2 / n2)) > 0 else 1e-8
    t_stat = (m1 - m2) / se
    
    # Welch-Satterthwaite degrees of freedom
    df_num = ((v1 / n1) + (v2 / n2)) ** 2
    df_den = (((v1 / n1) ** 2) / (n1 - 1)) + (((v2 / n2) ** 2) / (n2 - 1)) if n1 > 1 and n2 > 1 else 1.0
    df = df_num / df_den if df_den > 0 else 1.0
    
    # Approximate normal CDF tail for two-sided p-value
    abs_t = abs(t_stat)
    p_val = 2.0 * (1.0 - (0.5 * (1.0 + math.erf(abs_t / math.sqrt(2.0)))))
    
    return {
        "t_statistic": round(t_stat, 4),
        "df": round(df, 2),
        "p_value": round(p_val, 6),
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m

    for k, (original_idx, p_val) in enumerate(indexed_p):
        adjusted_alpha = alpha / (m - k)
        if p_val <= adjusted_alpha:
            results[original_idx] = True
        else:
            break

    return [(p_values[i], results[i]) for i in range(m)]

