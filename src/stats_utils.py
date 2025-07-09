import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style='whitegrid')

### --- DESCRIPTIVE STATS & VISUALS ---

def describe_numeric_column(df, column):
    """
    Print and return descriptive statistics for a numeric column.
    """
    desc = df[column].describe()
    print(f"\nDescriptive statistics for {column}:\n{desc}")
    return desc

def plot_distribution(df, column):
    """
    Plot histogram and KDE for a numeric column.
    """
    plt.figure(figsize=(8,4))
    sns.histplot(df[column], kde=True, color='teal', bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot_by_target(df, column, target):
    """
    Boxplot of a numeric column grouped by target variable (e.g., Churn).
    """
    plt.figure(figsize=(6,4))
    sns.boxplot(x=target, y=column, data=df, palette='pastel')
    plt.title(f'{column} by {target}')
    plt.show()

### --- CENTRAL LIMIT THEOREM (CLT) SAMPLING ---

def clt_sampling_distribution(df, column, sample_size=50, n_samples=1000, plot=True):
    """
    Demonstrate CLT: draw many samples and plot mean distribution.
    """
    sample_means = []
    for _ in range(n_samples):
        sample = df[column].dropna().sample(sample_size, replace=True)
        sample_means.append(sample.mean())
    
    if plot:
        plt.figure(figsize=(8,4))
        sns.histplot(sample_means, kde=True, color='skyblue', bins=30)
        plt.title(f'CLT Sampling Distribution of {column} Means (n={sample_size})')
        plt.xlabel('Sample Mean')
        plt.ylabel('Frequency')
        plt.show()
    
        _ = sample_means  # keep variable if you need internally

### --- CONFIDENCE INTERVALS & BOOTSTRAPPING ---

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for mean.
    """
    mean = np.mean(data)
    sem = stats.sem(data, nan_policy='omit')
    n = len(data)
    margin = sem * stats.t.ppf((1+confidence)/2., n-1)
    lower, upper = mean - margin, mean + margin
    print(f"{int(confidence*100)}% CI for mean: ({lower:.2f}, {upper:.2f})")
    return (lower, upper)

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95, plot=True):
    """
    Bootstrap confidence interval for mean.
    """
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data.dropna(), size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1-confidence)/2*100)
    upper = np.percentile(boot_means, (1+(confidence))/2*100)
    
    if plot:
        plt.figure(figsize=(8,4))
        sns.histplot(boot_means, kde=True, color='orchid', bins=30)
        plt.title(f'Bootstrap Mean Distribution ({n_bootstrap} resamples)')
        plt.axvline(lower, color='red', linestyle='--', label=f'{int(confidence*100)}% CI Lower')
        plt.axvline(upper, color='red', linestyle='--', label='Upper')
        plt.xlabel('Mean')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    print(f"{int(confidence*100)}% bootstrap CI: ({lower:.2f}, {upper:.2f})")
    return (lower, upper)

### --- HYPOTHESIS TESTS ---

def one_sample_ttest(data, popmean):
    """
    One-sample t-test against population mean.
    """
    t_stat, p_value = stats.ttest_1samp(data, popmean, nan_policy='omit')
    print(f"One-sample t-test: t={t_stat:.3f}, p={p_value:.3f}")
    return t_stat, p_value

def two_sample_ttest(data1, data2, equal_var=False):
    """
    Independent two-sample t-test.
    """
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, nan_policy='omit')
    print(f"Two-sample t-test: t={t_stat:.3f}, p={p_value:.3f}")
    return t_stat, p_value

def chi_squared_test(contingency_table):
    """
    Chi-squared test for independence.
    contingency_table: pd.DataFrame or array-like
    """
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-squared test: chi2={chi2:.3f}, dof={dof}, p={p:.3f}")
    return chi2, p, dof, expected

def anova_test(*groups):
    """
    One-way ANOVA to compare means across multiple groups.
    """
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
    return f_stat, p_value

### --- Z-SCORE ---

def calculate_z_score(value, mean, std):
    """
    Calculate z-score for given value.
    """
    z = (value - mean) / std
    print(f"Z-score: {z:.3f}")
    return z
