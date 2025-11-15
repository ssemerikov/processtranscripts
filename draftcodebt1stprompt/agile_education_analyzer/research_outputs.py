"""
Research-Ready Outputs Module
Generate LaTeX tables, statistical reports, and extract quotations for academic papers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import timedelta

from .data_structures import TranscriptSegment, SessionAnalysisResult
from .utils.logger import get_logger

logger = get_logger('research_outputs')

class ResearchOutputGenerator:
    """
    Generate publication-ready research outputs.

    Creates:
    - LaTeX tables formatted for academic papers
    - Statistical test result summaries
    - Quotation extraction with proper attribution
    - APA-formatted result sections
    """

    def __init__(self):
        """Initialize output generator"""
        logger.info("Research output generator initialized")

    def generate_latex_table(self, data: pd.DataFrame, caption: str,
                            label: str, format_spec: Optional[Dict] = None) -> str:
        """
        Generate LaTeX table from DataFrame.

        Args:
            data: DataFrame to convert
            caption: Table caption
            label: LaTeX label for cross-referencing
            format_spec: Optional formatting specifications

        Returns:
            LaTeX table string
        """
        logger.info(f"Generating LaTeX table: {label}")

        # Default format
        if format_spec is None:
            format_spec = {
                'position': 'htbp',
                'column_format': None,  # Auto-detect
                'float_format': '%.3f'
            }

        # Auto-detect column format if not specified
        if format_spec['column_format'] is None:
            n_cols = len(data.columns)
            format_spec['column_format'] = 'l' + 'c' * (n_cols - 1)

        # Start table
        latex = [
            r'\begin{table}[' + format_spec['position'] + ']',
            r'\centering',
            r'\caption{' + caption + '}',
            r'\label{tab:' + label + '}',
            r'\begin{tabular}{' + format_spec['column_format'] + '}',
            r'\toprule'
        ]

        # Header
        header = ' & '.join([self._escape_latex(str(col)) for col in data.columns])
        latex.append(header + r' \\')
        latex.append(r'\midrule')

        # Rows
        for _, row in data.iterrows():
            row_str = ' & '.join([
                self._format_latex_cell(val, format_spec['float_format'])
                for val in row
            ])
            latex.append(row_str + r' \\')

        # End table
        latex.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}'
        ])

        return '\n'.join(latex)

    def generate_descriptive_stats_table(self, results: List[SessionAnalysisResult],
                                        metric: str, by_sprint: bool = True) -> str:
        """
        Generate LaTeX table of descriptive statistics.

        Args:
            results: List of session analysis results
            metric: Metric to summarize
            by_sprint: Whether to group by sprint

        Returns:
            LaTeX table string
        """
        logger.info(f"Generating descriptive statistics table for {metric}")

        # Extract data
        data_rows = []
        for result in results:
            if by_sprint and result.metadata.sprint_number:
                sprint = result.metadata.sprint_number
                value = self._extract_metric_value(result, metric)
                if value is not None:
                    data_rows.append({'Sprint': sprint, metric: value})

        if not data_rows:
            logger.warning("No data for descriptive statistics table")
            return ""

        df = pd.DataFrame(data_rows)

        # Calculate statistics by sprint
        stats = df.groupby('Sprint')[metric].agg([
            ('N', 'count'),
            ('M', 'mean'),
            ('SD', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Median', 'median')
        ]).reset_index()

        caption = f"Descriptive Statistics for {metric} by Sprint"
        label = f"descriptive_{metric.replace(' ', '_').lower()}"

        return self.generate_latex_table(stats, caption, label)

    def generate_statistical_test_table(self, test_results: Dict) -> str:
        """
        Generate LaTeX table for statistical test results.

        Args:
            test_results: Dictionary with test results from StatisticalAnalyzer

        Returns:
            LaTeX table string
        """
        logger.info("Generating statistical test results table")

        rows = []

        # Extract relevant statistics
        if 'kruskal_wallis' in test_results:
            kw = test_results['kruskal_wallis']
            rows.append({
                'Test': 'Kruskal-Wallis H',
                'Statistic': f"H = {kw['H_statistic']:.3f}",
                'df': kw['degrees_of_freedom'],
                'p-value': f"{kw['p_value']:.4f}",
                'Result': 'Significant' if kw['significant'] else 'Not Significant'
            })

        if 'mann_whitney' in test_results:
            mw = test_results['mann_whitney']
            rows.append({
                'Test': 'Mann-Whitney U',
                'Statistic': f"U = {mw['U_statistic']:.3f}",
                'df': '-',
                'p-value': f"{mw['p_value']:.4f}",
                'Result': 'Significant' if mw['significant'] else 'Not Significant'
            })

        if 'effect_size' in test_results:
            es = test_results['effect_size']
            rows.append({
                'Test': "Cohen's d (Effect Size)",
                'Statistic': f"d = {es['cohens_d']:.3f}",
                'df': '-',
                'p-value': '-',
                'Result': es['interpretation'].capitalize()
            })

        if not rows:
            return ""

        df = pd.DataFrame(rows)
        caption = "Statistical Test Results"
        label = "statistical_tests"

        return self.generate_latex_table(df, caption, label)

    def generate_correlation_table(self, correlations: List[Dict]) -> str:
        """
        Generate correlation matrix as LaTeX table.

        Args:
            correlations: List of correlation results

        Returns:
            LaTeX table string
        """
        logger.info("Generating correlation table")

        rows = []
        for corr in correlations:
            if 'spearman' in corr:
                sp = corr['spearman']
                rows.append({
                    'Variable 1': corr['variables'][0],
                    'Variable 2': corr['variables'][1],
                    'Spearman ρ': sp['rho'],
                    'p-value': sp['p_value'],
                    'Significance': '*' if sp['significant'] else 'ns',
                    'Strength': sp['strength'].capitalize()
                })

        if not rows:
            return ""

        df = pd.DataFrame(rows)
        caption = "Correlation Analysis Results (Spearman's ρ)"
        label = "correlations"

        format_spec = {
            'position': 'htbp',
            'column_format': 'llcccc',
            'float_format': '%.3f'
        }

        latex_table = self.generate_latex_table(df, caption, label, format_spec)

        # Add note about significance
        note = (r'\begin{tablenotes}' + '\n' +
                r'\small' + '\n' +
                r'\item Note: * indicates p < .05, ns = not significant' + '\n' +
                r'\end{tablenotes}')

        # Insert note before \end{table}
        latex_table = latex_table.replace(r'\end{table}',
                                          note + '\n' + r'\end{table}')

        return latex_table

    def extract_quotations(self, segments: List[TranscriptSegment],
                          criteria: str = 'insightful',
                          max_quotes: int = 10) -> List[Dict]:
        """
        Extract relevant quotations from transcripts.

        Args:
            segments: List of transcript segments
            criteria: Selection criteria ('insightful', 'confused', 'questions')
            max_quotes: Maximum number of quotes to extract

        Returns:
            List of quotations with metadata
        """
        logger.info(f"Extracting quotations with criteria: {criteria}")

        quotations = []

        for segment in segments:
            should_include = False

            if criteria == 'insightful':
                # Long, detailed responses
                if (segment.text_length > 100 and
                    not segment.is_confusion and
                    segment.speaker_role == 'student'):
                    should_include = True

            elif criteria == 'confused':
                if segment.is_confusion:
                    should_include = True

            elif criteria == 'questions':
                if segment.is_question and segment.speaker_role == 'student':
                    should_include = True

            if should_include:
                quotations.append({
                    'text': segment.text,
                    'speaker': self._anonymize_speaker(segment.speaker),
                    'speaker_role': segment.speaker_role,
                    'timestamp': self._format_timestamp(segment.start_time),
                    'context': criteria,
                    'segment_index': segment.index
                })

        # Limit to max_quotes
        if len(quotations) > max_quotes:
            quotations = quotations[:max_quotes]

        logger.info(f"Extracted {len(quotations)} quotations")
        return quotations

    def format_quotation_latex(self, quotation: Dict) -> str:
        """
        Format a quotation in LaTeX for academic paper.

        Args:
            quotation: Quotation dictionary

        Returns:
            LaTeX-formatted quotation
        """
        text = self._escape_latex(quotation['text'])
        speaker = quotation.get('speaker', 'Participant')
        timestamp = quotation.get('timestamp', '')

        latex = (
            r'\begin{quote}' + '\n' +
            r'\textit{"' + text + r'"}' + '\n' +
            r'\end{quote}' + '\n' +
            r'\noindent --- ' + speaker + f' ({timestamp})\n'
        )

        return latex

    def generate_results_section(self, results_summary: Dict) -> str:
        """
        Generate APA-formatted results section text.

        Args:
            results_summary: Dictionary with analysis results

        Returns:
            Markdown-formatted results section
        """
        logger.info("Generating results section")

        sections = []

        sections.append("# Results\n")

        # Descriptive Statistics
        if 'descriptive_stats' in results_summary:
            sections.append("## Descriptive Statistics\n")
            sections.append(self._format_descriptive_results(
                results_summary['descriptive_stats']
            ))

        # Hypothesis Testing
        if 'hypothesis_tests' in results_summary:
            sections.append("## Hypothesis Testing\n")
            sections.append(self._format_hypothesis_results(
                results_summary['hypothesis_tests']
            ))

        # Correlations
        if 'correlations' in results_summary:
            sections.append("## Correlational Analysis\n")
            sections.append(self._format_correlation_results(
                results_summary['correlations']
            ))

        return '\n\n'.join(sections)

    def _extract_metric_value(self, result: SessionAnalysisResult, metric: str) -> Optional[float]:
        """Extract metric value from session result"""
        if metric in ['participation_rate', 'question_count']:
            return getattr(result.engagement_metrics, metric.replace('_rate', '_rate'), None)
        elif metric in ['total_agile_terms', 'student_adoption_rate']:
            return getattr(result.agile_metrics, metric, None)
        return None

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        # IMPORTANT: Backslash must be escaped FIRST to avoid double-escaping
        text = text.replace('\\', r'\textbackslash{}')

        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _format_latex_cell(self, value, float_format: str) -> str:
        """Format a cell value for LaTeX"""
        if pd.isna(value):
            return '-'
        elif isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, (float, np.floating)):
            return float_format % value
        else:
            return self._escape_latex(str(value))

    def _anonymize_speaker(self, speaker: Optional[str]) -> str:
        """Anonymize speaker for ethical reporting"""
        if not speaker:
            return "Participant"
        elif speaker.lower().startswith('student'):
            # Replace with generic identifier
            return "Student"
        elif speaker.lower().startswith('teacher'):
            return "Instructor"
        return "Participant"

    def _format_timestamp(self, timestamp: timedelta) -> str:
        """Format timestamp as MM:SS"""
        total_seconds = int(timestamp.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _format_descriptive_results(self, stats: Dict) -> str:
        """Format descriptive statistics in APA style"""
        text = []

        for metric, values in stats.items():
            text.append(
                f"For {metric}, *M* = {values['mean']:.2f}, "
                f"*SD* = {values['std']:.2f}, "
                f"range = [{values['min']:.2f}, {values['max']:.2f}]."
            )

        return ' '.join(text)

    def _format_hypothesis_results(self, tests: List[Dict]) -> str:
        """Format hypothesis test results in APA style"""
        text = []

        for test in tests:
            if 'kruskal_wallis' in test:
                kw = test['kruskal_wallis']
                text.append(
                    f"A Kruskal-Wallis test revealed "
                    f"{'a significant' if kw['significant'] else 'no significant'} "
                    f"difference across sprints, "
                    f"*H*({kw['degrees_of_freedom']}) = {kw['H_statistic']:.2f}, "
                    f"*p* = {kw['p_value']:.3f}."
                )

        return ' '.join(text)

    def _format_correlation_results(self, correlations: List[Dict]) -> str:
        """Format correlation results in APA style"""
        text = []

        for corr in correlations:
            if 'spearman' in corr:
                sp = corr['spearman']
                var1, var2 = corr['variables']
                text.append(
                    f"There was a {sp['strength']} "
                    f"{'significant' if sp['significant'] else 'non-significant'} "
                    f"correlation between {var1} and {var2}, "
                    f"*r<sub>s</sub>* = {sp['rho']:.2f}, "
                    f"*p* = {sp['p_value']:.3f}."
                )

        return ' '.join(text)

    def save_latex_document(self, tables: List[str], output_path: str):
        """
        Save complete LaTeX document with all tables.

        Args:
            tables: List of LaTeX table strings
            output_path: Output file path
        """
        document = [
            r'\documentclass[12pt]{article}',
            r'\usepackage{booktabs}',
            r'\usepackage{threeparttable}',
            r'\usepackage[utf8]{inputenc}',
            r'\usepackage[T2A]{fontenc}',  # For Cyrillic
            r'\usepackage[ukrainian,english]{babel}',
            r'',
            r'\begin{document}',
            r''
        ]

        document.extend(tables)

        document.append(r'\end{document}')

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(document))

        logger.info(f"LaTeX document saved to {output_path}")
