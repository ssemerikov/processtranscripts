"""
Logging Configuration for Agile Education Analysis Framework
Provides consistent logging across all modules with Ukrainian text support
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = 'agile_education_analyzer',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logger with proper formatting and handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        # Ensure proper UTF-8 encoding for Ukrainian text
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # More detailed in files
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger for a module.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f'agile_education_analyzer.{name}')

class ResearchLogger:
    """
    Specialized logger for research operations with context tracking.
    Helps maintain reproducibility by logging all analysis decisions.
    """

    def __init__(self, logger_name: str = 'research'):
        self.logger = get_logger(logger_name)
        self.analysis_log = []

    def log_parameter(self, param_name: str, param_value: any):
        """Log analysis parameter for reproducibility"""
        message = f"Parameter: {param_name} = {param_value}"
        self.logger.info(message)
        self.analysis_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'parameter',
            'name': param_name,
            'value': param_value
        })

    def log_decision(self, decision: str, rationale: str):
        """Log methodological decision"""
        message = f"Decision: {decision} | Rationale: {rationale}"
        self.logger.info(message)
        self.analysis_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'decision',
            'decision': decision,
            'rationale': rationale
        })

    def log_finding(self, finding: str, evidence: str):
        """Log research finding with evidence"""
        message = f"Finding: {finding} | Evidence: {evidence}"
        self.logger.info(message)
        self.analysis_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'finding',
            'finding': finding,
            'evidence': evidence
        })

    def log_ukrainian_text(self, context: str, text: str):
        """Log Ukrainian text with context (for validation)"""
        self.logger.debug(f"{context}: {text[:100]}..." if len(text) > 100 else f"{context}: {text}")

    def export_analysis_log(self, filepath: str):
        """Export analysis log for reproducibility documentation"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_log, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Analysis log exported to {filepath}")

# Create default logger
default_logger = setup_logger()
