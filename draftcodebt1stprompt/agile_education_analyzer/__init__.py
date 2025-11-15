"""
Agile Education Analysis Framework
===================================

A comprehensive framework for analyzing Ukrainian-language educational transcripts
from agile web programming courses.

Key Features:
- VTT transcript parsing with Ukrainian text handling
- Speaker diarization (teacher/student identification)
- Ukrainian discourse pattern detection
- Engagement and participation metrics
- Agile terminology adoption tracking
- Statistical analysis with proper research rigor
- Publication-ready visualizations
- Research outputs (LaTeX tables, quotations)

Quick Start:
-----------
```python
from agile_education_analyzer import AgileEducationAnalyzer

# Initialize analyzer
analyzer = AgileEducationAnalyzer('path/to/transcripts')

# Run complete analysis
results = analyzer.analyze_all_sessions()

# Generate visualizations
analyzer.generate_visualizations(results, output_dir='./outputs')

# Export results
analyzer.export_results(results, format='latex')
```

Modules:
--------
- data_structures: Core data models
- ukrainian_patterns: Ukrainian discourse detection
- visualization: Publication-ready charts
- statistical_analysis: Rigorous statistical testing
- research_outputs: LaTeX tables and quotations
- utils: Logging and helper functions
"""

__version__ = '1.0.0'
__author__ = 'Agile Education Research Team'
__license__ = 'MIT'

# Core data structures
from .data_structures import (
    TranscriptSegment,
    SessionMetadata,
    UkrainianDiscoursePattern,
    AgileTerminology,
    ProblemInstance,
    EngagementMetrics,
    AgileAdoptionMetrics,
    TeachingPattern,
    Insight,
    SessionAnalysisResult,
    SpeakerRole,
    DiscourseType
)

# Analysis modules
from .ukrainian_patterns import (
    UkrainianDiscourseDetector,
    DialoguePatternAnalyzer
)

from .statistical_analysis import StatisticalAnalyzer
from .visualization import ResearchVisualizer
from .research_outputs import ResearchOutputGenerator

# Utilities
from .utils.logger import setup_logger, get_logger, ResearchLogger

__all__ = [
    # Version info
    '__version__',

    # Data structures
    'TranscriptSegment',
    'SessionMetadata',
    'UkrainianDiscoursePattern',
    'AgileTerminology',
    'ProblemInstance',
    'EngagementMetrics',
    'AgileAdoptionMetrics',
    'TeachingPattern',
    'Insight',
    'SessionAnalysisResult',
    'SpeakerRole',
    'DiscourseType',

    # Analysis classes
    'UkrainianDiscourseDetector',
    'DialoguePatternAnalyzer',
    'StatisticalAnalyzer',
    'ResearchVisualizer',
    'ResearchOutputGenerator',

    # Utilities
    'setup_logger',
    'get_logger',
    'ResearchLogger',
]
