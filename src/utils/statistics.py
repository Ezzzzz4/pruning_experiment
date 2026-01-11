"""
Statistical Utilities

Functions for computing confidence intervals, significance tests, and effect sizes.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: List of observations
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    arr = np.array(values)
    n = len(arr)
    
    if n < 2:
        mean = np.mean(arr)
        return (mean, mean)
    
    mean = np.mean(arr)
    std_err = stats.sem(arr)
    
    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std_err
    
    return (mean - margin, mean + margin)


def significance_test(
    group_a: List[float],
    group_b: List[float],
    test: str = 't-test'
) -> Dict[str, float]:
    """
    Perform statistical significance test between two groups.
    
    Args:
        group_a: First group of observations
        group_b: Second group of observations
        test: Type of test ('t-test' or 'mann-whitney')
        
    Returns:
        Dict with statistic, p-value, and significance
    """
    a = np.array(group_a)
    b = np.array(group_b)
    
    if test == 't-test':
        statistic, p_value = stats.ttest_ind(a, b)
    elif test == 'mann-whitney':
        statistic, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'significant_001': p_value < 0.001,
    }


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    Args:
        group_a: First group
        group_b: Second group
        
    Returns:
        Cohen's d value
    """
    a = np.array(group_a)
    b = np.array(group_b)
    
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(a) - np.mean(b)) / pooled_std


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    elif d_abs < 0.5:
        return 'small'
    elif d_abs < 0.8:
        return 'medium'
    else:
        return 'large'


def compute_stats_summary(values: List[float]) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a list of values.
    
    Args:
        values: List of observations
        
    Returns:
        Dictionary with mean, std, median, min, max, ci_95
    """
    arr = np.array(values)
    ci_low, ci_high = compute_confidence_interval(values)
    
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'n': len(arr),
    }


def correlation_analysis(
    x: List[float],
    y: List[float]
) -> Dict[str, float]:
    """
    Compute correlation between two variables.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Dict with Pearson and Spearman correlations
    """
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    pearson_r, pearson_p = stats.pearsonr(x_arr, y_arr)
    spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
    }
