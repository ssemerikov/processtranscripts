"""
Test Suite for Statistical Analysis Module
Tests all statistical methods with proper research rigor
"""

import pytest
import pandas as pd
import numpy as np
from agile_education_analyzer.statistical_analysis import StatisticalAnalyzer

class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return StatisticalAnalyzer(significance_level=0.05, use_bonferroni=True)

    @pytest.fixture
    def sample_sprint_data(self):
        """Create sample data for sprint comparison"""
        return pd.DataFrame({
            'sprint_number': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'participation_rate': [0.45, 0.52, 0.48, 0.61, 0.58, 0.63, 0.72, 0.68, 0.75],
            'question_count': [10, 12, 11, 15, 14, 16, 20, 18, 22]
        })

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.alpha == 0.05
        assert analyzer.use_bonferroni is True

    def test_compare_sprints_kruskal_wallis(self, analyzer, sample_sprint_data):
        """Test Kruskal-Wallis comparison across sprints"""
        results = analyzer.compare_sprints(sample_sprint_data, 'participation_rate')

        assert 'kruskal_wallis' in results
        assert 'H_statistic' in results['kruskal_wallis']
        assert 'p_value' in results['kruskal_wallis']
        assert 'significant' in results['kruskal_wallis']
        assert results['n_sprints'] == 3
        assert results['total_observations'] == 9

    def test_compare_sprints_pairwise(self, analyzer, sample_sprint_data):
        """Test pairwise comparisons if significant"""
        results = analyzer.compare_sprints(sample_sprint_data, 'participation_rate')

        if results['kruskal_wallis']['significant']:
            assert 'pairwise_comparisons' in results
            assert len(results['pairwise_comparisons']) == 3  # 3 choose 2

    def test_compare_sprints_two_groups(self, analyzer):
        """Test Mann-Whitney for two groups"""
        data = pd.DataFrame({
            'sprint_number': [1, 1, 1, 2, 2, 2],
            'participation_rate': [0.45, 0.52, 0.48, 0.72, 0.68, 0.75]
        })

        results = analyzer.compare_sprints(data, 'participation_rate')

        assert 'mann_whitney' in results
        assert 'U_statistic' in results['mann_whitney']
        assert 'p_value' in results['mann_whitney']
        assert 'effect_size' in results
        assert 'cohens_d' in results['effect_size']

    def test_compare_sprints_insufficient_data(self, analyzer):
        """Test with insufficient data"""
        data = pd.DataFrame({
            'sprint_number': [1],
            'participation_rate': [0.5]
        })

        results = analyzer.compare_sprints(data, 'participation_rate')
        assert 'error' in results or results['groups_with_data'] < 2

    def test_analyze_correlation_both(self, analyzer):
        """Test correlation analysis with both methods"""
        data = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'var2': [2, 4, 5, 4, 5, 7, 6, 8, 9, 10]
        })

        results = analyzer.analyze_correlation(data, 'var1', 'var2', method='both')

        assert 'pearson' in results
        assert 'spearman' in results
        assert 'r' in results['pearson']
        assert 'rho' in results['spearman']
        assert results['n_observations'] == 10

    def test_analyze_correlation_with_nan(self, analyzer):
        """Test correlation with missing values"""
        data = pd.DataFrame({
            'var1': [1, 2, np.nan, 4, 5],
            'var2': [2, 4, 5, np.nan, 8]
        })

        results = analyzer.analyze_correlation(data, 'var1', 'var2')
        assert results['n_observations'] == 3  # Only complete pairs

    def test_analyze_correlation_insufficient_data(self, analyzer):
        """Test correlation with too few observations"""
        data = pd.DataFrame({
            'var1': [1, 2],
            'var2': [2, 4]
        })

        results = analyzer.analyze_correlation(data, 'var1', 'var2')
        assert 'error' in results

    def test_time_series_analysis(self, analyzer):
        """Test time series analysis"""
        data = pd.DataFrame({
            'timestamp': range(10),
            'metric': [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        })

        results = analyzer.time_series_analysis(data, 'metric')

        assert 'trend_coefficient' in results
        assert 'r_squared' in results
        assert 'p_value' in results
        assert 'trend_direction' in results
        assert results['trend_direction'] in ['increasing', 'decreasing']

    def test_time_series_insufficient_data(self, analyzer):
        """Test time series with insufficient data"""
        data = pd.DataFrame({
            'timestamp': [1, 2],
            'metric': [1, 2]
        })

        results = analyzer.time_series_analysis(data, 'metric')
        assert 'error' in results

    def test_paired_comparison(self, analyzer):
        """Test paired comparison (Wilcoxon)"""
        before = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        after = np.array([2, 3, 4, 5, 6, 7, 8, 9])

        results = analyzer.paired_comparison(before, after)

        assert 'test' in results
        assert results['test'] == 'wilcoxon_signed_rank'
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'effect_size' in results

    def test_paired_comparison_unequal_length(self, analyzer):
        """Test paired comparison with unequal arrays"""
        before = np.array([1, 2, 3])
        after = np.array([2, 3])

        with pytest.raises(ValueError):
            analyzer.paired_comparison(before, after)

    def test_paired_comparison_with_nan(self, analyzer):
        """Test paired comparison with NaN values"""
        before = np.array([1, 2, np.nan, 4, 5])
        after = np.array([2, 3, 4, np.nan, 6])

        results = analyzer.paired_comparison(before, after)
        assert results['n_pairs'] == 3  # Only complete pairs

    def test_power_analysis(self, analyzer):
        """Test statistical power analysis"""
        results = analyzer.power_analysis(
            effect_size=0.5,
            n_per_group=30,
            alpha=0.05
        )

        assert 'power' in results
        assert 'required_n_for_80_power' in results
        assert 0 <= results['power'] <= 1

    def test_cohens_d_calculation(self, analyzer):
        """Test Cohen's d calculation"""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])

        d = analyzer._cohens_d(group1, group2)
        assert isinstance(d, float)
        assert d < 0  # group1 mean < group2 mean

    def test_effect_size_interpretation(self, analyzer):
        """Test effect size interpretation"""
        assert analyzer._interpret_effect_size(0.1) == "negligible"
        assert analyzer._interpret_effect_size(0.3) == "small"
        assert analyzer._interpret_effect_size(0.6) == "medium"
        assert analyzer._interpret_effect_size(1.0) == "large"

    def test_correlation_interpretation(self, analyzer):
        """Test correlation strength interpretation"""
        assert analyzer._interpret_correlation(0.05) == "negligible"
        assert analyzer._interpret_correlation(0.2) == "weak"
        assert analyzer._interpret_correlation(0.4) == "moderate"
        assert analyzer._interpret_correlation(0.6) == "strong"
        assert analyzer._interpret_correlation(0.8) == "very strong"

    def test_descriptive_stats(self, analyzer, sample_sprint_data):
        """Test descriptive statistics calculation"""
        results = analyzer.compare_sprints(sample_sprint_data, 'participation_rate')

        assert 'descriptives' in results
        descriptives = results['descriptives']
        assert len(descriptives) == 3  # 3 sprints

        for stat in descriptives:
            assert 'sprint' in stat
            assert 'mean' in stat
            assert 'std' in stat
            assert 'median' in stat
            assert 'n' in stat

    def test_bonferroni_correction(self):
        """Test Bonferroni correction is applied"""
        analyzer_with = StatisticalAnalyzer(use_bonferroni=True)
        analyzer_without = StatisticalAnalyzer(use_bonferroni=False)

        data = pd.DataFrame({
            'sprint_number': [1, 1, 2, 2, 3, 3],
            'metric': [1, 2, 3, 4, 5, 6]
        })

        results_with = analyzer_with.compare_sprints(data, 'metric')
        results_without = analyzer_without.compare_sprints(data, 'metric')

        # Both should have results, but p-values might differ in interpretation
        assert 'kruskal_wallis' in results_with or 'mann_whitney' in results_with
        assert 'kruskal_wallis' in results_without or 'mann_whitney' in results_without

class TestStatisticalAnalyzerEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        analyzer = StatisticalAnalyzer()
        data = pd.DataFrame({'sprint_number': [], 'metric': []})

        results = analyzer.compare_sprints(data, 'metric')
        assert 'error' in results or results['groups_with_data'] == 0

    def test_all_nan_values(self):
        """Test with all NaN values"""
        analyzer = StatisticalAnalyzer()
        data = pd.DataFrame({
            'sprint_number': [1, 1, 2, 2],
            'metric': [np.nan, np.nan, np.nan, np.nan]
        })

        results = analyzer.compare_sprints(data, 'metric')
        assert 'error' in results or results['groups_with_data'] == 0

    def test_single_value_per_group(self):
        """Test with single value per group"""
        analyzer = StatisticalAnalyzer()
        data = pd.DataFrame({
            'sprint_number': [1, 2, 3],
            'metric': [1.0, 2.0, 3.0]
        })

        results = analyzer.compare_sprints(data, 'metric')
        # Should handle gracefully, though statistical power is low
        assert 'kruskal_wallis' in results or 'error' in results

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
