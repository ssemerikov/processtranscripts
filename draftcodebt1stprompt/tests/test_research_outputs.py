"""
Test Suite for Research Outputs Module
Tests LaTeX generation, quotation extraction, and APA formatting
"""

import pytest
import pandas as pd
from datetime import timedelta

from agile_education_analyzer.research_outputs import ResearchOutputGenerator
from agile_education_analyzer.data_structures import TranscriptSegment

class TestResearchOutputGenerator:
    """Test ResearchOutputGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create output generator instance"""
        return ResearchOutputGenerator()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'Sprint': [1, 2, 3],
            'Mean': [0.45, 0.61, 0.72],
            'SD': [0.05, 0.04, 0.06],
            'N': [10, 10, 10]
        })

    def test_generator_initialization(self, generator):
        """Test generator initialization"""
        assert generator is not None

    def test_escape_latex_special_characters(self, generator):
        """Test LaTeX special character escaping"""
        text = "Test & % $ # _ { } ~ ^"
        escaped = generator._escape_latex(text)

        assert r'\&' in escaped
        assert r'\%' in escaped
        assert r'\$' in escaped
        assert r'\#' in escaped
        assert r'\_' in escaped

    def test_escape_latex_ukrainian(self, generator):
        """Test that Ukrainian text is not corrupted"""
        text = "Привіт"
        escaped = generator._escape_latex(text)
        # Ukrainian should pass through unchanged (no special chars)
        assert "Привіт" in escaped

    def test_generate_latex_table_basic(self, generator, sample_dataframe):
        """Test basic LaTeX table generation"""
        latex = generator.generate_latex_table(
            sample_dataframe,
            caption="Test Table",
            label="test"
        )

        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex
        assert r'\caption{Test Table}' in latex
        assert r'\label{tab:test}' in latex
        assert r'\toprule' in latex
        assert r'\bottomrule' in latex

    def test_generate_latex_table_with_format(self, generator, sample_dataframe):
        """Test LaTeX table with custom format"""
        format_spec = {
            'position': 'h',
            'column_format': 'lccc',
            'float_format': '%.2f'
        }

        latex = generator.generate_latex_table(
            sample_dataframe,
            caption="Formatted Table",
            label="formatted",
            format_spec=format_spec
        )

        assert '[h]' in latex
        assert '{lccc}' in latex

    def test_format_latex_cell_integer(self, generator):
        """Test formatting integer cells"""
        result = generator._format_latex_cell(42, '%.3f')
        assert result == '42'

    def test_format_latex_cell_float(self, generator):
        """Test formatting float cells"""
        result = generator._format_latex_cell(3.14159, '%.2f')
        assert result == '3.14'

    def test_format_latex_cell_nan(self, generator):
        """Test formatting NaN cells"""
        import numpy as np
        result = generator._format_latex_cell(np.nan, '%.3f')
        assert result == '-'

    def test_format_latex_cell_string(self, generator):
        """Test formatting string cells"""
        result = generator._format_latex_cell("test_string", '%.3f')
        assert "test" in result

    def test_anonymize_speaker_student(self, generator):
        """Test speaker anonymization for students"""
        assert generator._anonymize_speaker("Student_1") == "Student"
        assert generator._anonymize_speaker("Student_5") == "Student"

    def test_anonymize_speaker_teacher(self, generator):
        """Test speaker anonymization for teacher"""
        assert generator._anonymize_speaker("Teacher") == "Instructor"
        assert generator._anonymize_speaker("teacher_name") == "Instructor"

    def test_anonymize_speaker_none(self, generator):
        """Test anonymization of None speaker"""
        assert generator._anonymize_speaker(None) == "Participant"

    def test_format_timestamp(self, generator):
        """Test timestamp formatting"""
        timestamp = timedelta(minutes=5, seconds=30)
        formatted = generator._format_timestamp(timestamp)
        assert formatted == "05:30"

    def test_format_timestamp_hours(self, generator):
        """Test timestamp with hours"""
        timestamp = timedelta(hours=1, minutes=15, seconds=45)
        formatted = generator._format_timestamp(timestamp)
        # Should show as minutes:seconds (75:45)
        assert "75:45" in formatted or "1:15:45" in formatted

    def test_extract_quotations_questions(self, generator):
        """Test extracting question quotations"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=5),
                text="Що таке спринт?",
                speaker="Student_1",
                speaker_role="student",
                is_question=True
            ),
            TranscriptSegment(
                index=1,
                start_time=timedelta(seconds=6),
                end_time=timedelta(seconds=10),
                text="Не питання",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=False
            )
        ]

        quotations = generator.extract_quotations(segments, criteria='questions', max_quotes=10)

        assert len(quotations) >= 1
        assert quotations[0]['text'] == "Що таке спринт?"
        assert quotations[0]['speaker'] == "Student"

    def test_extract_quotations_confused(self, generator):
        """Test extracting confusion quotations"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=5),
                text="Не розумію як це працює",
                speaker="Student_1",
                speaker_role="student",
                is_confusion=True
            )
        ]

        quotations = generator.extract_quotations(segments, criteria='confused', max_quotes=10)

        assert len(quotations) >= 1
        assert quotations[0]['context'] == 'confused'

    def test_extract_quotations_max_limit(self, generator):
        """Test quotation extraction respects max_quotes"""
        segments = [
            TranscriptSegment(
                index=i,
                start_time=timedelta(seconds=i*5),
                end_time=timedelta(seconds=(i+1)*5),
                text=f"Question {i}?",
                speaker=f"Student_{i}",
                speaker_role="student",
                is_question=True
            )
            for i in range(20)
        ]

        quotations = generator.extract_quotations(segments, criteria='questions', max_quotes=5)
        assert len(quotations) == 5

    def test_format_quotation_latex(self, generator):
        """Test formatting a quotation in LaTeX"""
        quotation = {
            'text': 'Що таке спринт?',
            'speaker': 'Student',
            'timestamp': '05:30'
        }

        latex = generator.format_quotation_latex(quotation)

        assert r'\begin{quote}' in latex
        assert r'\end{quote}' in latex
        assert r'\textit{' in latex
        assert 'Що таке спринт?' in latex
        assert 'Student' in latex
        assert '05:30' in latex

    def test_generate_statistical_test_table(self, generator):
        """Test generating statistical test results table"""
        test_results = {
            'kruskal_wallis': {
                'H_statistic': 15.234,
                'p_value': 0.0012,
                'significant': True,
                'degrees_of_freedom': 2
            },
            'effect_size': {
                'cohens_d': 0.65,
                'interpretation': 'medium'
            }
        }

        latex = generator.generate_statistical_test_table(test_results)

        assert 'Kruskal-Wallis' in latex
        assert 'Cohen' in latex or 'Effect Size' in latex
        assert '15.234' in latex or '15.23' in latex

    def test_generate_statistical_test_table_mann_whitney(self, generator):
        """Test generating Mann-Whitney test table"""
        test_results = {
            'mann_whitney': {
                'U_statistic': 45.5,
                'p_value': 0.032,
                'significant': True
            }
        }

        latex = generator.generate_statistical_test_table(test_results)

        assert 'Mann-Whitney' in latex

    def test_generate_statistical_test_table_empty(self, generator):
        """Test with no statistical results"""
        test_results = {}

        latex = generator.generate_statistical_test_table(test_results)

        assert latex == ""

class TestResearchOutputEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def generator(self):
        """Create output generator instance"""
        return ResearchOutputGenerator()

    def test_latex_table_empty_dataframe(self, generator):
        """Test LaTeX generation with empty DataFrame"""
        df = pd.DataFrame()

        latex = generator.generate_latex_table(df, "Empty", "empty")

        # Should generate table structure even if empty
        assert r'\begin{table}' in latex

    def test_latex_with_unicode_characters(self, generator):
        """Test LaTeX generation with Unicode characters"""
        df = pd.DataFrame({
            'Metric': ['Участь', 'Питання'],
            'Value': [0.75, 15]
        })

        latex = generator.generate_latex_table(df, "Ukrainian Table", "ukrainian")

        # Ukrainian characters should be preserved
        assert 'Участь' in latex or 'Metric' in latex

    def test_extract_quotations_empty_list(self, generator):
        """Test quotation extraction with empty segment list"""
        quotations = generator.extract_quotations([], criteria='questions')

        assert quotations == []

    def test_extract_quotations_no_matches(self, generator):
        """Test quotation extraction with no matching criteria"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=5),
                text="Statement",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=False,
                is_confusion=False
            )
        ]

        quotations = generator.extract_quotations(segments, criteria='questions')

        assert len(quotations) == 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
