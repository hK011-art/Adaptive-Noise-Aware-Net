import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

def perform_paired_comparison(
    baseline_metrics: List[float],
    proposed_metrics: List[float],
    metric_name: str = "PSNR",
    confidence_level: float = 0.95,
    hypothesis_family: str = "Primary benchmark comparison"
) -> Dict[str, Any]:
    """
    Performs rigorous paired statistical evaluation between two capacity-matched models
    (e.g. Model D vs Model E) evaluated on identical sample manifests.
    
    Computes:
    - Mean difference and Median difference (proposed - baseline)
    - Win rate (% samples where proposed > baseline)
    - Paired effect size: Cohen's dz = mean(diff) / std(diff)
    - Paired Student-t 95% Confidence Interval for mean difference
    - Paired Wilcoxon signed-rank test p-value (non-parametric)
    
    Parameters:
        baseline_metrics: Evaluation metric values for baseline model across samples.
        proposed_metrics: Evaluation metric values for proposed model across the same samples.
        metric_name: Name of metric (e.g. 'PSNR', 'SSIM', 'LPIPS').
        confidence_level: Confidence level for CI (default 0.95 for 95% CI).
        hypothesis_family: Logical test group (e.g. 'Primary benchmark comparison', 'Ablation study').
    """
    assert len(baseline_metrics) == len(proposed_metrics), "Sample counts must match exactly for paired testing!"
    n = len(baseline_metrics)
    if n == 0:
        return {}

    b_arr = np.array(baseline_metrics, dtype=np.float64)
    p_arr = np.array(proposed_metrics, dtype=np.float64)
    diffs = p_arr - b_arr

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    win_rate = float(np.sum(diffs > 0) / n)

    # Paired Cohen's dz effect size: mean(D) / std(D, ddof=1)
    diff_std = float(np.std(diffs, ddof=1)) if n > 1 else 1e-8
    cohens_dz = mean_diff / diff_std if diff_std > 1e-12 else 0.0

    # Paired Student-t Confidence Interval
    se_diff = diff_std / np.sqrt(n)
    try:
        from scipy import stats
        alpha_ci = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df=n - 1)) if n > 1 else 1.96
    except Exception:
        t_crit = 1.96
    ci_margin = float(t_crit * se_diff)
    ci_lower = mean_diff - ci_margin
    ci_upper = mean_diff + ci_margin

    # Wilcoxon signed-rank test (two-sided)
    p_value = float('nan')
    is_significant = False
    try:
        from scipy import stats
        if np.all(diffs == 0):
            p_value = 1.0
        else:
            # zero_method='wilcox' discards zero differences per standard methodology
            res = stats.wilcoxon(p_arr, b_arr, zero_method='wilcox', alternative='two-sided')
            p_value = float(res.pvalue)
            is_significant = (p_value < 0.05)
    except Exception as e:
        print(f"[Stats Note] scipy.stats.wilcoxon unavailable or errored: {e}")

    result = {
        "metric": metric_name,
        "hypothesis_family": hypothesis_family,
        "n_samples": n,
        "mean_diff": mean_diff,
        "mean_difference": mean_diff,
        "median_diff": median_diff,
        "win_rate": win_rate,
        "cohens_d": cohens_dz,
        "cohens_dz": cohens_dz,
        "effect_size_definition": "Cohen's dz = mean(D) / std(D, ddof=1) for paired differences D = proposed - baseline",
        "ci_unit_of_analysis": "image-level paired differences (proposed - baseline)",
        "ci_method": "paired Student-t distribution",
        "ci_confidence_level": confidence_level,
        "mean_diff_ci_lower": ci_lower,
        "mean_diff_ci_upper": ci_upper,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_diff_ci_margin": ci_margin,
        "p_value": p_value,
        "wilcoxon_p_value": p_value,
        "is_significant_p05": is_significant,
        "is_statistically_significant": is_significant,
        "interpretation": "statistically significant" if is_significant else "not statistically significant"
    }
    return result


def apply_holm_bonferroni_correction(
    results_list: List[Dict[str, Any]],
    alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Applies Holm-Bonferroni stepwise correction for multiple simultaneous hypothesis tests
    (e.g. comparing Model E across multiple baselines or ISO levels).
    Controls Family-Wise Error Rate (FWER) without inflating Type II errors.
    
    Guarantees strict monotonicity of adjusted p-values via cumulative maximum:
        p_adj(k) = min(1.0, max_{j <= k} (m - j + 1) * p_(j))
    
    Restores original sample/result order before returning.
    """
    if not results_list:
        return []

    m = len(results_list)
    # Extract raw p-values with original index preservation
    indexed_pvals = []
    for idx, r in enumerate(results_list):
        p = r.get("p_value", 1.0)
        indexed_pvals.append((idx, p if not np.isnan(p) else 1.0))

    # Sort strictly by raw p-value
    indexed_pvals.sort(key=lambda x: x[1])

    corrected_results = [dict(r) for r in results_list]
    cummax = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_pvals):
        k = rank + 1
        raw_adj = p_val * (m - k + 1)
        # Cumulative maximum ensures non-decreasing monotonicity
        cummax = max(cummax, raw_adj)
        p_adj = min(cummax, 1.0)
        threshold = alpha / (m - k + 1)
        
        # Step-down rejection: rejected if and only if p_adj <= alpha
        is_sig_holm = (p_adj <= alpha)
        
        corrected_results[orig_idx]["holm_rank"] = k
        corrected_results[orig_idx]["holm_threshold"] = float(threshold)
        corrected_results[orig_idx]["p_value_holm_adjusted"] = float(p_adj)
        corrected_results[orig_idx]["is_significant_holm"] = is_sig_holm

    return corrected_results


def export_statistical_report(results: List[Dict[str, Any]], output_path: str = "statistical_tests.csv"):
    """Exports paired comparison table to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[Statistical Testing] Exported paired test results to {output_path}")
