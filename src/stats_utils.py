import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Union

def descriptive_statistics(series: pd.Series) -> dict:
    """
    Calculate mean, median, mode, std, and variance for a numeric pandas Series.
    """
    return {
        'mean': series.mean(),
        'median': series.median(),
        'mode': series.mode().iloc[0] if not series.mode().empty else np.nan,
        'std': series.std(),
        'variance': series.var()
    }

def confidence_interval_mean(series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean of a numeric pandas Series.
    """
    n = series.dropna().shape[0]
    mean = series.mean()
    stderr = stats.sem(series, nan_policy='omit')
    h = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

def perform_ttest(series1: pd.Series, series2: pd.Series, equal_var: bool = False) -> Tuple[float, float]:
    """
    Perform two-sample t-test.
    Returns: t-statistic, p-value
    """
    t_stat, p_val = stats.ttest_ind(series1, series2, equal_var=equal_var, nan_policy='omit')
    return t_stat, p_val

def perform_anova(*groups: List[pd.Series]) -> Tuple[float, float]:
    """
    Perform one-way ANOVA test on multiple groups.
    Returns: F-statistic, p-value
    """
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def perform_chi2_test(contingency_table: pd.DataFrame) -> Tuple[float, float, int, np.ndarray]:
    """
    Perform Chi-squared test of independence.
    contingency_table: typically from pd.crosstab.
    Returns: chi2-statistic, p-value, degrees of freedom, expected frequencies
    """
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p, dof, expected

def central_limit_theorem_sampling(series: pd.Series, sample_size: int, n_samples: int = 1000) -> np.ndarray:
    """
    Demonstrate CLT by taking many samples and computing sample means.
    Returns: array of sample means.
    """
    sample_means = [
        series.sample(sample_size, replace=True).mean()
        for _ in range(n_samples)
    ]
    return np.array(sample_means)

def bootstrap_confidence_interval(series: pd.Series, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Use bootstrap sampling to compute confidence interval for the mean.
    """
    boot_means = [
        series.sample(frac=1, replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return lower, upper

def calculate_z_score(
    value_or_series: Union[float, pd.Series],
    mean: float = None,
    std: float = None
) -> Union[float, pd.Series]:
    """
    Calculate z-score for a single value or for a Series.
    If mean and std are None and input is Series, will use Series mean and std.
    """
    if isinstance(value_or_series, pd.Series):
        mu = mean if mean is not None else value_or_series.mean()
        sigma = std if std is not None else value_or_series.std()
        return (value_or_series - mu) / sigma
    else:
        if mean is None or std is None:
            raise ValueError("For single value, mean and std must be provided.")
        return (value_or_series - mean) / std
