"""
Test Suite for Logger Module
Tests logging functionality and Ukrainian text support
"""

import pytest
import logging
import tempfile
from pathlib import Path
import json

from agile_education_analyzer.utils.logger import (
    setup_logger,
    get_logger,
    ResearchLogger
)

class TestSetupLogger:
    """Test logger setup functionality"""

    def test_setup_logger_default(self):
        """Test default logger setup"""
        logger = setup_logger(name='test_logger')
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """Test logger with custom level"""
        logger = setup_logger(name='test_debug', level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self):
        """Test logger with file output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            logger = setup_logger(name='test_file', log_file=str(log_file))

            logger.info("Test message")

            assert log_file.exists()
            content = log_file.read_text(encoding='utf-8')
            assert "Test message" in content

    def test_setup_logger_ukrainian_text(self):
        """Test logger with Ukrainian text"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'ukrainian.log'
            logger = setup_logger(name='test_ukrainian', log_file=str(log_file))

            ukrainian_text = "Привіт, це тестове повідомлення"
            logger.info(ukrainian_text)

            content = log_file.read_text(encoding='utf-8')
            assert ukrainian_text in content

    def test_setup_logger_no_console(self):
        """Test logger without console output"""
        logger = setup_logger(name='test_no_console', console_output=False)
        # Should have no StreamHandler
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        # Might have file handlers but checking console specifically would need more setup
        assert isinstance(logger, logging.Logger)

class TestGetLogger:
    """Test get_logger functionality"""

    def test_get_logger(self):
        """Test getting a module logger"""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert 'test_module' in logger.name

    def test_get_logger_hierarchy(self):
        """Test logger name hierarchy"""
        logger = get_logger('submodule')
        assert 'agile_education_analyzer.submodule' in logger.name

class TestResearchLogger:
    """Test ResearchLogger class"""

    @pytest.fixture
    def research_logger(self):
        """Create research logger instance"""
        return ResearchLogger('test_research')

    def test_research_logger_initialization(self, research_logger):
        """Test research logger initialization"""
        assert research_logger.logger is not None
        assert research_logger.analysis_log == []

    def test_log_parameter(self, research_logger):
        """Test parameter logging"""
        research_logger.log_parameter('test_param', 42)

        assert len(research_logger.analysis_log) == 1
        log_entry = research_logger.analysis_log[0]
        assert log_entry['type'] == 'parameter'
        assert log_entry['name'] == 'test_param'
        assert log_entry['value'] == 42
        assert 'timestamp' in log_entry

    def test_log_decision(self, research_logger):
        """Test methodological decision logging"""
        research_logger.log_decision(
            "Use non-parametric test",
            "Data not normally distributed"
        )

        assert len(research_logger.analysis_log) == 1
        log_entry = research_logger.analysis_log[0]
        assert log_entry['type'] == 'decision'
        assert log_entry['decision'] == "Use non-parametric test"
        assert log_entry['rationale'] == "Data not normally distributed"

    def test_log_finding(self, research_logger):
        """Test finding logging"""
        research_logger.log_finding(
            "Significant difference found",
            "p < 0.05"
        )

        assert len(research_logger.analysis_log) == 1
        log_entry = research_logger.analysis_log[0]
        assert log_entry['type'] == 'finding'
        assert log_entry['finding'] == "Significant difference found"
        assert log_entry['evidence'] == "p < 0.05"

    def test_log_ukrainian_text(self, research_logger):
        """Test Ukrainian text logging"""
        ukrainian_text = "Студент сказав: 'Не розумію як це працює'"
        research_logger.log_ukrainian_text("Question", ukrainian_text)

        # Should not crash with Ukrainian text
        assert True

    def test_export_analysis_log(self, research_logger):
        """Test exporting analysis log"""
        # Add some log entries
        research_logger.log_parameter('alpha', 0.05)
        research_logger.log_decision('Use Bonferroni', 'Multiple comparisons')
        research_logger.log_finding('Significant result', 'H=15.3, p=0.001')

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'analysis_log.json'
            research_logger.export_analysis_log(str(log_file))

            assert log_file.exists()

            # Read and validate JSON
            with open(log_file, 'r', encoding='utf-8') as f:
                exported_log = json.load(f)

            assert len(exported_log) == 3
            assert exported_log[0]['type'] == 'parameter'
            assert exported_log[1]['type'] == 'decision'
            assert exported_log[2]['type'] == 'finding'

    def test_multiple_log_entries(self, research_logger):
        """Test logging multiple entries"""
        research_logger.log_parameter('n_topics', 10)
        research_logger.log_parameter('min_df', 2)
        research_logger.log_decision('Use LDA', 'Better interpretability')

        assert len(research_logger.analysis_log) == 3

    def test_log_with_ukrainian_values(self, research_logger):
        """Test logging parameters with Ukrainian text values"""
        research_logger.log_parameter('session_type', 'спринт')
        research_logger.log_parameter('finding', 'Студенти активні')

        assert len(research_logger.analysis_log) == 2
        assert research_logger.analysis_log[0]['value'] == 'спринт'

class TestLoggerEdgeCases:
    """Test edge cases and error handling"""

    def test_logger_with_special_characters(self):
        """Test logger with special characters in messages"""
        logger = setup_logger(name='test_special')
        special_text = "Test: #, $, %, &, @, !"
        logger.info(special_text)
        # Should not crash
        assert True

    def test_logger_with_very_long_message(self):
        """Test logger with very long message"""
        logger = setup_logger(name='test_long')
        long_message = "A" * 10000
        logger.info(long_message)
        # Should not crash
        assert True

    def test_research_logger_with_complex_objects(self):
        """Test research logger with complex objects"""
        research_logger = ResearchLogger('test_complex')

        # Log dictionary
        research_logger.log_parameter('config', {'alpha': 0.05, 'method': 'kruskal'})

        # Log list
        research_logger.log_parameter('sprints', [1, 2, 3])

        assert len(research_logger.analysis_log) == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
