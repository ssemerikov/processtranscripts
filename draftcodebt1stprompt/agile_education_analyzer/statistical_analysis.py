"""
Statistical Analysis Module for Educational Research
Rigorous statistical testing appropriate for educational data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    mannwhitneyu, kruskal, friedmanchisquare,
    spearmanr, kendalltau, chi2_contingency,
    shapiro, levene
)
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
from typing import Dict, List, Tuple, Optional
import warnings

from .utils.logger import get_logger, ResearchLogger

logger = get_logger('statistical_analysis')
research_logger = ResearchLogger('statistics')

class StatisticalAnalyzer:
    """
    Statistical analysis for educational research with proper rigor.

    Implements non-parametric tests appropriate for educational data,
    calculates effect sizes, applies multiple comparison corrections,
    and maintains reproducibility through logging.
    """

    def __init__(self, significance_level: float = 0.05,
                 use_bonferroni: bool = True):
        """
        Initialize statistical analyzer.

        Args:
            significance_level: Alpha level for hypothesis testing
            use_bonferroni: Whether to apply Bonferroni correction
        """
        self.alpha = significance_level
        self.use_bonferroni = use_bonferroni

        research_logger.log_parameter('significance_level', significance_level)
        research_logger.log_parameter('bonferroni_correction', use_bonferroni)

        logger.info(f"Statistical analyzer initialized: α={significance_level}, "
                   f"Bonferroni={use_bonferroni}")

    def compare_sprints(self, sprint_data: pd.DataFrame, metric: str) -> Dict:
        """
        Compare metrics across sprints using appropriate statistical tests.

        Uses Kruskal-Wallis H-test (non-parametric) for multiple groups,
        followed by pairwise Mann-Whitney U tests with Bonferroni correction.

        Args:
            sprint_data: DataFrame with sprint_number and metric columns
            metric: Name of metric to compare

        Returns:
            Dictionary with test results including statistics, p-values,
            effect sizes, and post-hoc comparisons
        """
        logger.info(f"Comparing {metric} across sprints")

        results = {
            'metric': metric,
            'n_sprints': sprint_data['sprint_number'].nunique(),
            'total_observations': len(sprint_data)
        }

        # Group data by sprint
        sprint_groups = [
            group[metric].dropna().values
            for _, group in sprint_data.groupby('sprint_number')
        ]

        # Filter out empty groups
        sprint_groups = [g for g in sprint_groups if len(g) > 0]
        results['groups_with_data'] = len(sprint_groups)

        if len(sprint_groups) < 2:
            logger.warning(f"Insufficient groups for comparison ({len(sprint_groups)})")
            results['error'] = "Insufficient groups for comparison"
            return results

        # Check assumptions
        results['assumptions'] = self._check_assumptions(sprint_groups)

        # Kruskal-Wallis test for multiple groups
        if len(sprint_groups) > 2:
            h_stat, p_value = kruskal(*sprint_groups)
            results['kruskal_wallis'] = {
                'H_statistic': float(h_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'degrees_of_freedom': len(sprint_groups) - 1
            }

            logger.info(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_value:.4f}")
            research_logger.log_finding(
                f"Kruskal-Wallis test for {metric}",
                f"H={h_stat:.4f}, p={p_value:.4f}, significant={p_value < self.alpha}"
            )

            # Post-hoc pairwise comparisons if significant
            if p_value < self.alpha:
                results['pairwise_comparisons'] = self._pairwise_comparisons(
                    sprint_groups, sprint_data
                )

        # For two groups, use Mann-Whitney U
        elif len(sprint_groups) == 2:
            u_stat, p_value = mannwhitneyu(sprint_groups[0], sprint_groups[1],
                                           alternative='two-sided')
            results['mann_whitney'] = {
                'U_statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha
            }

            # Effect size (Cohen's d)
            cohens_d = self._cohens_d(sprint_groups[0], sprint_groups[1])
            results['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_effect_size(cohens_d)
            }

            logger.info(f"Mann-Whitney U: U={u_stat:.4f}, p={p_value:.4f}, d={cohens_d:.4f}")

        # Descriptive statistics by group
        results['descriptives'] = self._descriptive_stats(sprint_data, metric)

        return results

    def analyze_correlation(self, data: pd.DataFrame, var1: str, var2: str,
                           method: str = 'both') -> Dict:
        """
        Analyze correlation between variables.

        Args:
            data: DataFrame with variables
            var1: First variable name
            var2: Second variable name
            method: 'pearson', 'spearman', 'kendall', or 'both'

        Returns:
            Dictionary with correlation coefficients and significance tests
        """
        logger.info(f"Analyzing correlation: {var1} vs {var2}")

        # Remove NaN values
        clean_data = data[[var1, var2]].dropna()

        if len(clean_data) < 3:
            logger.warning("Insufficient data for correlation")
            return {'error': 'Insufficient data'}

        results = {
            'variables': (var1, var2),
            'n_observations': len(clean_data)
        }

        # Pearson correlation (parametric)
        if method in ['pearson', 'both']:
            pearson_r, pearson_p = stats.pearsonr(clean_data[var1], clean_data[var2])
            results['pearson'] = {
                'r': float(pearson_r),
                'r_squared': float(pearson_r ** 2),
                'p_value': float(pearson_p),
                'significant': pearson_p < self.alpha,
                'strength': self._interpret_correlation(pearson_r)
            }

        # Spearman correlation (non-parametric)
        if method in ['spearman', 'both']:
            spearman_r, spearman_p = spearmanr(clean_data[var1], clean_data[var2])
            results['spearman'] = {
                'rho': float(spearman_r),
                'p_value': float(spearman_p),
                'significant': spearman_p < self.alpha,
                'strength': self._interpret_correlation(spearman_r)
            }

        # Kendall tau (for ordinal data)
        if method == 'kendall':
            kendall_tau, kendall_p = kendalltau(clean_data[var1], clean_data[var2])
            results['kendall'] = {
                'tau': float(kendall_tau),
                'p_value': float(kendall_p),
                'significant': kendall_p < self.alpha
            }

        logger.info(f"Correlation results: {results}")
        return results

    def time_series_analysis(self, data: pd.DataFrame, metric: str,
                            time_column: str = 'timestamp') -> Dict:
        """
        Analyze trends over time using regression.

        Args:
            data: DataFrame with time series data
            metric: Metric to analyze
            time_column: Name of time/sequence column

        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Time series analysis for {metric}")

        # Sort by time
        data_sorted = data.sort_values(time_column).copy()
        data_sorted = data_sorted[[time_column, metric]].dropna()

        if len(data_sorted) < 3:
            logger.warning("Insufficient data for time series analysis")
            return {'error': 'Insufficient data'}

        # Create sequential index if time_column is datetime
        X = np.arange(len(data_sorted)).reshape(-1, 1)
        y = data_sorted[metric].values

        # Linear regression for trend
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        results = {
            'metric': metric,
            'n_observations': len(data_sorted),
            'trend_coefficient': float(model.params[1]),
            'intercept': float(model.params[0]),
            'r_squared': float(model.rsquared),
            'adjusted_r_squared': float(model.rsquared_adj),
            'p_value': float(model.pvalues[1]),
            'confidence_interval_95': [float(x) for x in model.conf_int()[1].tolist()],
            'trend_direction': 'increasing' if model.params[1] > 0 else 'decreasing',
            'trend_significant': model.pvalues[1] < self.alpha
        }

        # Additional metrics
        results['f_statistic'] = float(model.fvalue)
        results['f_pvalue'] = float(model.f_pvalue)

        logger.info(f"Trend: {results['trend_direction']}, "
                   f"β={results['trend_coefficient']:.4f}, "
                   f"p={results['p_value']:.4f}")

        research_logger.log_finding(
            f"Time trend for {metric}",
            f"Direction={results['trend_direction']}, R²={results['r_squared']:.3f}"
        )

        return results

    def paired_comparison(self, before: np.ndarray, after: np.ndarray) -> Dict:
        """
        Compare paired observations (e.g., pre-post intervention).

        Uses Wilcoxon signed-rank test (non-parametric paired test).

        Args:
            before: Measurements before intervention
            after: Measurements after intervention

        Returns:
            Dictionary with test results and effect size
        """
        logger.info("Performing paired comparison")

        if len(before) != len(after):
            raise ValueError("Arrays must have same length for paired comparison")

        # Remove pairs with NaN
        mask = ~(np.isnan(before) | np.isnan(after))
        before = before[mask]
        after = after[mask]

        if len(before) < 3:
            return {'error': 'Insufficient paired observations'}

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(before, after)

        # Effect size (Cohen's d for paired data)
        differences = after - before
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0

        results = {
            'test': 'wilcoxon_signed_rank',
            'n_pairs': len(before),
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'mean_difference': float(mean_diff),
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_effect_size(cohens_d)
            }
        }

        logger.info(f"Paired comparison: W={statistic:.4f}, p={p_value:.4f}, d={cohens_d:.4f}")
        return results

    def power_analysis(self, effect_size: float, n_per_group: int,
                      alpha: float = None) -> Dict:
        """
        Calculate statistical power for t-test.

        Args:
            effect_size: Expected Cohen's d
            n_per_group: Sample size per group
            alpha: Significance level (uses self.alpha if None)

        Returns:
            Dictionary with power analysis results
        """
        alpha = alpha or self.alpha

        power_analysis = TTestIndPower()
        power = power_analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            ratio=1.0
        )

        # Calculate required sample size for 80% power
        required_n = power_analysis.solve_power(
            effect_size=effect_size,
            power=0.80,
            alpha=alpha,
            ratio=1.0
        )

        results = {
            'effect_size': effect_size,
            'n_per_group': n_per_group,
            'alpha': alpha,
            'power': float(power),
            'required_n_for_80_power': int(np.ceil(required_n))
        }

        logger.info(f"Power analysis: power={power:.3f}, required_n={required_n:.0f}")
        return results

    def _pairwise_comparisons(self, sprint_groups: List[np.ndarray],
                             sprint_data: pd.DataFrame) -> List[Dict]:
        """Perform pairwise Mann-Whitney U tests with Bonferroni correction"""
        comparisons = []
        n_comparisons = len(sprint_groups) * (len(sprint_groups) - 1) // 2

        adjusted_alpha = self.alpha / n_comparisons if self.use_bonferroni else self.alpha

        for i in range(len(sprint_groups)):
            for j in range(i + 1, len(sprint_groups)):
                u_stat, p_val = mannwhitneyu(sprint_groups[i], sprint_groups[j],
                                             alternative='two-sided')

                cohens_d = self._cohens_d(sprint_groups[i], sprint_groups[j])

                comparisons.append({
                    'sprint_pair': (i + 1, j + 1),
                    'u_statistic': float(u_stat),
                    'p_value': float(p_val),
                    'p_value_adjusted': float(p_val * n_comparisons) if self.use_bonferroni else float(p_val),
                    'significant': p_val < adjusted_alpha,
                    'cohens_d': float(cohens_d),
                    'effect_interpretation': self._interpret_effect_size(cohens_d)
                })

        return comparisons

    def _check_assumptions(self, groups: List[np.ndarray]) -> Dict:
        """Check statistical assumptions for the data"""
        assumptions = {}

        # Normality test (Shapiro-Wilk) for each group
        normality_results = []
        for i, group in enumerate(groups):
            if len(group) >= 3:
                stat, p = shapiro(group)
                normality_results.append({
                    'group': i + 1,
                    'is_normal': p > 0.05,
                    'p_value': float(p)
                })

        assumptions['normality'] = normality_results
        assumptions['all_normal'] = all(r['is_normal'] for r in normality_results)

        # Homogeneity of variance (Levene's test)
        if len(groups) >= 2:
            stat, p = levene(*groups)
            assumptions['homogeneity_of_variance'] = {
                'equal_variances': p > 0.05,
                'p_value': float(p)
            }

        return assumptions

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        r_abs = abs(r)
        if r_abs < 0.1:
            return "negligible"
        elif r_abs < 0.3:
            return "weak"
        elif r_abs < 0.5:
            return "moderate"
        elif r_abs < 0.7:
            return "strong"
        else:
            return "very strong"

    def _descriptive_stats(self, data: pd.DataFrame, metric: str) -> List[Dict]:
        """Calculate descriptive statistics by group"""
        descriptives = []

        for sprint_num, group in data.groupby('sprint_number'):
            values = group[metric].dropna()

            if len(values) > 0:
                descriptives.append({
                    'sprint': int(sprint_num),
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values, ddof=1)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q1': float(np.percentile(values, 25)),
                    'q3': float(np.percentile(values, 75)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
                })

        return descriptives

    def export_analysis_log(self, filepath: str):
        """Export complete analysis log for reproducibility"""
        research_logger.export_analysis_log(filepath)
        logger.info(f"Analysis log exported to {filepath}")
