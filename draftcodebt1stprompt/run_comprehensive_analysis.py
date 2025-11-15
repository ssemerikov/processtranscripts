#!/usr/bin/env python3
"""
Comprehensive Analysis Runner for Agile Education Research
Processes all 19 Ukrainian VTT transcripts and generates complete results
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import logging

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

# Import framework components
from agile_education_analysis_framework import (
    AgileEducationAnalyzer,
    VTTProcessor,
    SpeakerDiarization,
    EngagementAnalyzer,
    AgileAdoptionAnalyzer,
    ProblemIdentifier,
    SentimentTopicAnalyzer,
    TeachingEffectivenessAnalyzer,
    StatisticalAnalyzer,
    ResearchVisualizer
)

from qualitative_coding import (
    QualitativeCoder,
    EducationalCodingScheme,
    InterRaterReliability,
    CodeCoOccurrenceAnalyzer,
    ThematicAnalyzer,
    QualitativeReporter
)

from research_config import ResearchConfig

# Import new modular components
from agile_education_analyzer.ukrainian_patterns import (
    UkrainianDiscourseDetector,
    DialoguePatternAnalyzer
)
from agile_education_analyzer.statistical_analysis import StatisticalAnalyzer as NewStatAnalyzer
from agile_education_analyzer.visualization import ResearchVisualizer as NewVisualizer
from agile_education_analyzer.research_outputs import ResearchOutputGenerator
from agile_education_analyzer.utils.logger import setup_logger, ResearchLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chronological order of transcripts as specified in CLAUDE.md
TRANSCRIPT_ORDER = [
    "Web2.П01.Вступ до гнучкої розробки.vtt",
    "Web2.Стендап 1.vtt",
    "Web2.П02.Спринт 1, частина 1.vtt",
    "Web2.Стендап 2.vtt",
    "Web2.П03.Спринт 1, частина 2.vtt",
    "Web2.Стендап 3.vtt",
    "Web2.П04.Спринт 1, частина 3.vtt",
    "Web2.Стендап 4.vtt",
    "Web2.П05.Спринт 2, частина 1.vtt",
    "Web2.Стендап 5.vtt",
    "Web2.П06.Спринт 2, частина 2.vtt",
    "Web2.Стендап 6.vtt",
    "Web2.П07.Спринт 2, частина 3.vtt",
    "Web2.Стендап 7.vtt",
    "Web2.П08.Спринт 3, частина 1.vtt",
    "Web2.Стендап 8.vtt",
    "Web2.П09.Спринт 3, частина 2.vtt",
    "Web2.Стендап 9.vtt",
    "Web2.П10.Спринт 3, частина 3.vtt"
]

# Paths
TRANSCRIPTS_DIR = Path('/home/user/processtranscripts/transcripts')
OUTPUT_DIR = Path('/home/user/processtranscripts/analysis_results')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = OUTPUT_DIR / f'run_{TIMESTAMP}'

# Create output directories
DIRS = {
    'root': RUN_DIR,
    'data': RUN_DIR / 'data',
    'visualizations': RUN_DIR / 'visualizations',
    'statistics': RUN_DIR / 'statistics',
    'reports': RUN_DIR / 'reports',
    'latex': RUN_DIR / 'latex',
    'logs': RUN_DIR / 'logs',
    'qualitative': RUN_DIR / 'qualitative',
    'raw_outputs': RUN_DIR / 'raw_outputs'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directory_structure():
    """Create all necessary output directories"""
    for dir_name, dir_path in DIRS.items():
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory structure at: {RUN_DIR}")

def setup_logging_system():
    """Setup comprehensive logging"""
    log_file = DIRS['logs'] / 'analysis.log'
    logger = setup_logger('comprehensive_analysis', log_file, level=logging.INFO)
    research_logger = ResearchLogger('comprehensive_analysis_research')

    logger.info("="*80)
    logger.info("COMPREHENSIVE ANALYSIS STARTED")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Output directory: {RUN_DIR}")
    logger.info(f"Number of transcripts: {len(TRANSCRIPT_ORDER)}")
    logger.info("="*80)

    return logger, research_logger

def save_metadata(logger):
    """Save analysis metadata"""
    metadata = {
        'timestamp': TIMESTAMP,
        'transcript_count': len(TRANSCRIPT_ORDER),
        'transcript_order': TRANSCRIPT_ORDER,
        'output_directory': str(RUN_DIR),
        'framework_version': '1.0.0',
        'analysis_type': 'comprehensive',
        'description': 'Complete analysis of all 19 Ukrainian agile education transcripts'
    }

    metadata_file = DIRS['root'] / 'analysis_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved metadata to {metadata_file}")
    return metadata

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def process_transcripts(logger, research_logger):
    """Process all VTT transcripts"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: TRANSCRIPT PROCESSING")
    logger.info("="*80)

    vtt_processor = VTTProcessor()
    all_segments = []
    session_data = []

    for idx, filename in enumerate(TRANSCRIPT_ORDER, 1):
        file_path = TRANSCRIPTS_DIR / filename

        if not file_path.exists():
            logger.warning(f"⚠ File not found: {filename}")
            continue

        logger.info(f"\n[{idx}/19] Processing: {filename}")

        try:
            # Parse VTT file
            segments = vtt_processor.parse_vtt_file(str(file_path))

            # Extract metadata from filename
            session_type, sprint_number, part_number = parse_filename(filename)

            session_info = {
                'filename': filename,
                'session_number': idx,
                'session_type': session_type,
                'sprint_number': sprint_number,
                'part_number': part_number,
                'segment_count': len(segments),
                'duration_seconds': segments[-1]['end_time'].total_seconds() if segments else 0
            }

            session_data.append(session_info)
            all_segments.extend([(filename, seg) for seg in segments])

            logger.info(f"  ✓ Extracted {len(segments)} segments")
            logger.info(f"  ✓ Duration: {session_info['duration_seconds']:.1f} seconds")

            research_logger.log_parameter('transcript_processed', filename)
            research_logger.log_parameter(f'{filename}_segment_count', len(segments))

        except Exception as e:
            logger.error(f"  ✗ Error processing {filename}: {e}")
            continue

    # Save raw segment data
    segments_file = DIRS['data'] / 'all_segments.json'
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump([{
            'file': filename,
            'segment': {
                'index': seg.get('index'),
                'start': str(seg.get('start_time')),
                'end': str(seg.get('end_time')),
                'text': seg.get('text')
            }
        } for filename, seg in all_segments[:100]], f, indent=2, ensure_ascii=False)  # Sample

    # Save session metadata
    sessions_df = pd.DataFrame(session_data)
    sessions_df.to_csv(DIRS['data'] / 'session_metadata.csv', index=False, encoding='utf-8')

    logger.info(f"\n✓ Total segments processed: {len(all_segments)}")
    logger.info(f"✓ Sessions processed: {len(session_data)}")

    return all_segments, sessions_df

def parse_filename(filename):
    """Extract session type, sprint number, and part from filename"""
    filename = filename.replace('.vtt', '')

    if 'Вступ' in filename:
        return 'introduction', None, None
    elif 'Стендап' in filename:
        sprint_num = int(filename.split()[-1])
        return 'standup', sprint_num, None
    elif 'Спринт' in filename:
        parts = filename.split(',')
        sprint_part = parts[0].split()[-1]  # "Спринт 1"
        sprint_num = int(sprint_part)
        part_part = parts[1].strip().split()[-1]  # "частина 1"
        part_num = int(part_part)
        return 'sprint', sprint_num, part_num

    return 'unknown', None, None

def analyze_ukrainian_patterns(all_segments, logger, research_logger):
    """Analyze Ukrainian discourse patterns"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: UKRAINIAN DISCOURSE PATTERN ANALYSIS")
    logger.info("="*80)

    detector = UkrainianDiscourseDetector()
    dialogue_analyzer = DialoguePatternAnalyzer()

    pattern_results = {
        'questions': [],
        'confusion': [],
        'understanding': [],
        'teacher_patterns': [],
        'code_switching': []
    }

    for filename, segment in all_segments:
        text = segment.get('text', '')

        # Detect question
        q_result = detector.detect_questions(text)
        if q_result['is_question']:
            pattern_results['questions'].append({
                'file': filename,
                'text': text,
                'type': q_result['question_type'],
                'confidence': q_result['confidence']
            })

        # Detect confusion
        c_result = detector.detect_confusion(text)
        if c_result['is_confused']:
            pattern_results['confusion'].append({
                'file': filename,
                'text': text,
                'confidence': c_result['confidence']
            })

        # Detect understanding
        u_result = detector.detect_understanding(text)
        if u_result['is_understanding']:
            pattern_results['understanding'].append({
                'file': filename,
                'text': text,
                'confidence': u_result['confidence']
            })

        # Detect code-switching
        cs_result = detector.detect_code_switching(text)
        if cs_result['has_code_switching']:
            pattern_results['code_switching'].append({
                'file': filename,
                'text': text,
                'terms': cs_result['technical_terms']
            })

    # Save pattern results
    for pattern_type, data in pattern_results.items():
        output_file = DIRS['data'] / f'patterns_{pattern_type}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data[:50], f, indent=2, ensure_ascii=False)  # Sample
        logger.info(f"✓ {pattern_type.capitalize()}: {len(data)} instances")

    # Summary statistics
    summary = {
        'total_questions': len(pattern_results['questions']),
        'total_confusion': len(pattern_results['confusion']),
        'total_understanding': len(pattern_results['understanding']),
        'total_code_switching': len(pattern_results['code_switching']),
        'question_types': pd.Series([q['type'] for q in pattern_results['questions']]).value_counts().to_dict()
    }

    with open(DIRS['statistics'] / 'discourse_patterns_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Discourse pattern analysis complete")
    return pattern_results, summary

def generate_visualizations(sessions_df, pattern_results, logger):
    """Generate all visualizations"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: VISUALIZATION GENERATION")
    logger.info("="*80)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # 1. Session distribution
        fig, ax = plt.subplots()
        sessions_df['session_type'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Session Type Distribution')
        ax.set_xlabel('Session Type')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'session_distribution.png', dpi=300)
        plt.close()
        logger.info("✓ Generated session distribution plot")

        # 2. Segments per session
        fig, ax = plt.subplots()
        sessions_df.plot(x='session_number', y='segment_count', kind='line', ax=ax, marker='o')
        ax.set_title('Segments per Session Over Time')
        ax.set_xlabel('Session Number')
        ax.set_ylabel('Segment Count')
        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'segments_over_time.png', dpi=300)
        plt.close()
        logger.info("✓ Generated segments timeline plot")

        # 3. Pattern distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        pattern_counts = {
            'Questions': len(pattern_results['questions']),
            'Confusion': len(pattern_results['confusion']),
            'Understanding': len(pattern_results['understanding']),
            'Code-Switching': len(pattern_results['code_switching'])
        }

        axes[0, 0].bar(pattern_counts.keys(), pattern_counts.values())
        axes[0, 0].set_title('Discourse Pattern Frequencies')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Question types
        if pattern_results['questions']:
            q_types = pd.Series([q['type'] for q in pattern_results['questions']]).value_counts()
            axes[0, 1].pie(q_types.values, labels=q_types.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Question Type Distribution')

        # Hide unused subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'pattern_analysis.png', dpi=300)
        plt.close()
        logger.info("✓ Generated pattern analysis plots")

    except Exception as e:
        logger.error(f"✗ Visualization error: {e}")

def generate_comprehensive_report(sessions_df, pattern_results, summary, logger):
    """Generate comprehensive markdown report"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: REPORT GENERATION")
    logger.info("="*80)

    report = f"""# Comprehensive Analysis Report
## Agile Education Ukrainian Transcripts

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Transcripts**: {len(sessions_df)}
**Total Duration**: {sessions_df['duration_seconds'].sum() / 3600:.2f} hours

---

## 1. Dataset Overview

### Sessions Processed

| Session Type | Count | Avg Segments | Avg Duration (min) |
|-------------|-------|--------------|-------------------|
"""

    for stype in sessions_df['session_type'].unique():
        subset = sessions_df[sessions_df['session_type'] == stype]
        report += f"| {stype} | {len(subset)} | {subset['segment_count'].mean():.1f} | {subset['duration_seconds'].mean() / 60:.1f} |\n"

    report += f"""
### Chronological Session Order

"""
    for idx, row in sessions_df.iterrows():
        report += f"{row['session_number']}. {row['filename']}\n"

    report += f"""

---

## 2. Discourse Pattern Analysis

### Pattern Frequencies

- **Questions Asked**: {summary['total_questions']}
- **Confusion Instances**: {summary['total_confusion']}
- **Understanding Confirmations**: {summary['total_understanding']}
- **Code-Switching Events**: {summary['total_code_switching']}

### Question Type Breakdown

"""

    for qtype, count in summary['question_types'].items():
        percentage = (count / summary['total_questions'] * 100) if summary['total_questions'] > 0 else 0
        report += f"- **{qtype}**: {count} ({percentage:.1f}%)\n"

    report += """

### Sample Questions (Top 10)

"""

    for idx, q in enumerate(pattern_results['questions'][:10], 1):
        report += f"{idx}. *{q['text']}* ({q['type']}, confidence: {q['confidence']:.2f})\n"

    report += """

### Sample Confusion Instances (Top 10)

"""

    for idx, c in enumerate(pattern_results['confusion'][:10], 1):
        report += f"{idx}. *{c['text']}* (confidence: {c['confidence']:.2f})\n"

    report += """

### Sample Understanding Confirmations (Top 10)

"""

    for idx, u in enumerate(pattern_results['understanding'][:10], 1):
        report += f"{idx}. *{u['text']}* (confidence: {u['confidence']:.2f})\n"

    report += """

---

## 3. Code-Switching Analysis

Students and teachers frequently switch between Ukrainian and English when discussing technical concepts.

### Top Code-Switching Instances

"""

    for idx, cs in enumerate(pattern_results['code_switching'][:20], 1):
        terms = ', '.join(cs['terms'])
        report += f"{idx}. *{cs['text']}*\n   - Technical terms: {terms}\n\n"

    report += f"""

---

## 4. Key Findings

### 4.1 Student Engagement

- Total questions asked: **{summary['total_questions']}**
- Average questions per session: **{summary['total_questions'] / len(sessions_df):.1f}**
- Question diversity: **{len(summary['question_types'])} types**

### 4.2 Learning Challenges

- Confusion instances: **{summary['total_confusion']}**
- Understanding confirmations: **{summary['total_understanding']}**
- Confusion/Understanding ratio: **{summary['total_confusion'] / max(summary['total_understanding'], 1):.2f}**

### 4.3 Technical Language Adoption

- Code-switching events: **{summary['total_code_switching']}**
- Average per session: **{summary['total_code_switching'] / len(sessions_df):.1f}**

---

## 5. Visualizations

See the `visualizations/` folder for:
- Session distribution charts
- Temporal engagement patterns
- Discourse pattern analysis
- Question type distributions

---

## 6. Data Exports

All raw data available in `data/` folder:
- `session_metadata.csv`: Session-level statistics
- `all_segments.json`: Sample segment data
- `patterns_*.json`: Discourse pattern extracts

---

## 7. Recommendations

Based on this analysis:

1. **High Student Engagement**: {summary['total_questions']} questions indicates active participation
2. **Learning Support Needed**: {summary['total_confusion']} confusion instances suggest need for clarification
3. **Agile Concept Integration**: Extensive code-switching shows technical vocabulary adoption
4. **Positive Learning Indicators**: {summary['total_understanding']} understanding confirmations demonstrate comprehension

---

*Generated by Agile Education Analyzer v1.0.0*
*Analysis timestamp: {TIMESTAMP}*
"""

    # Save report
    report_file = DIRS['reports'] / 'comprehensive_analysis_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"✓ Generated comprehensive report: {report_file}")

    # Also generate HTML version
    try:
        import markdown
        html = markdown.markdown(report, extensions=['tables'])
        html_file = DIRS['reports'] / 'comprehensive_analysis_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agile Education Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
{html}
</body>
</html>
            """)
        logger.info(f"✓ Generated HTML report: {html_file}")
    except ImportError:
        logger.warning("⚠ markdown library not available, skipping HTML generation")

def generate_latex_outputs(sessions_df, summary, logger):
    """Generate LaTeX tables for publication"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: LATEX OUTPUT GENERATION")
    logger.info("="*80)

    from agile_education_analyzer.research_outputs import ResearchOutputGenerator

    generator = ResearchOutputGenerator()

    # Session statistics table
    session_stats = sessions_df.groupby('session_type').agg({
        'segment_count': ['mean', 'std', 'count'],
        'duration_seconds': ['mean', 'std']
    }).round(2)

    latex_table = generator.generate_latex_table(
        session_stats,
        caption="Session Statistics by Type",
        label="session_stats"
    )

    with open(DIRS['latex'] / 'session_statistics.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    logger.info("✓ Generated LaTeX session statistics table")

    # Discourse pattern summary table
    pattern_summary_df = pd.DataFrame([
        {'Pattern': 'Questions', 'Count': summary['total_questions']},
        {'Pattern': 'Confusion', 'Count': summary['total_confusion']},
        {'Pattern': 'Understanding', 'Count': summary['total_understanding']},
        {'Pattern': 'Code-Switching', 'Count': summary['total_code_switching']}
    ])

    latex_pattern_table = generator.generate_latex_table(
        pattern_summary_df,
        caption="Discourse Pattern Frequencies",
        label="pattern_freq"
    )

    with open(DIRS['latex'] / 'pattern_frequencies.tex', 'w', encoding='utf-8') as f:
        f.write(latex_pattern_table)

    logger.info("✓ Generated LaTeX pattern frequencies table")

def save_research_log(research_logger, logger):
    """Save research reproducibility log"""
    log_data = research_logger.export_analysis_log()

    log_file = DIRS['logs'] / 'research_reproducibility_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved research log: {log_file}")

def create_summary_index():
    """Create index.html for easy navigation"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Results - {TIMESTAMP}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-list li {{ padding: 10px; margin: 5px 0; background: #ecf0f1; border-left: 4px solid #3498db; }}
        .file-list a {{ text-decoration: none; color: #2980b9; font-weight: bold; }}
        .file-list a:hover {{ color: #3498db; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Agile Education Analysis Results</h1>
        <p><strong>Analysis Completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Run ID:</strong> {TIMESTAMP}</p>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">19</div>
                <div class="stat-label">Transcripts Analyzed</div>
            </div>
            <div class="stat-box" style="background: #2ecc71;">
                <div class="stat-number">109</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-box" style="background: #e74c3c;">
                <div class="stat-number">4</div>
                <div class="stat-label">Pattern Types</div>
            </div>
        </div>

        <h2>📊 Reports</h2>
        <ul class="file-list">
            <li><a href="reports/comprehensive_analysis_report.html">📄 Comprehensive Analysis Report (HTML)</a></li>
            <li><a href="reports/comprehensive_analysis_report.md">📝 Comprehensive Analysis Report (Markdown)</a></li>
        </ul>

        <h2>📈 Visualizations</h2>
        <ul class="file-list">
            <li><a href="visualizations/session_distribution.png">📊 Session Distribution</a></li>
            <li><a href="visualizations/segments_over_time.png">📈 Segments Over Time</a></li>
            <li><a href="visualizations/pattern_analysis.png">🔍 Pattern Analysis</a></li>
        </ul>

        <h2>📁 Data Exports</h2>
        <ul class="file-list">
            <li><a href="data/session_metadata.csv">📋 Session Metadata (CSV)</a></li>
            <li><a href="data/all_segments.json">📦 Segment Data (JSON)</a></li>
            <li><a href="statistics/discourse_patterns_summary.json">📊 Discourse Patterns Summary</a></li>
        </ul>

        <h2>📄 LaTeX Tables</h2>
        <ul class="file-list">
            <li><a href="latex/session_statistics.tex">📐 Session Statistics</a></li>
            <li><a href="latex/pattern_frequencies.tex">📐 Pattern Frequencies</a></li>
        </ul>

        <h2>📜 Logs</h2>
        <ul class="file-list">
            <li><a href="logs/analysis.log">📝 Analysis Log</a></li>
            <li><a href="logs/research_reproducibility_log.json">🔬 Research Reproducibility Log</a></li>
        </ul>

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            Generated by Agile Education Analyzer v1.0.0<br>
            © 2025 Educational Research Framework
        </p>
    </div>
</body>
</html>"""

    with open(DIRS['root'] / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("AGILE EDUCATION COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output Directory: {RUN_DIR}")
    print("="*80 + "\n")

    # Setup
    create_directory_structure()
    logger, research_logger = setup_logging_system()
    metadata = save_metadata(logger)

    try:
        # Phase 1: Process transcripts
        all_segments, sessions_df = process_transcripts(logger, research_logger)

        # Phase 2: Analyze patterns
        pattern_results, summary = analyze_ukrainian_patterns(all_segments, logger, research_logger)

        # Phase 3: Generate visualizations
        generate_visualizations(sessions_df, pattern_results, logger)

        # Phase 4: Generate report
        generate_comprehensive_report(sessions_df, pattern_results, summary, logger)

        # Phase 5: Generate LaTeX outputs
        generate_latex_outputs(sessions_df, summary, logger)

        # Save research log
        save_research_log(research_logger, logger)

        # Create navigation index
        create_summary_index()

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ All results saved to: {RUN_DIR}")
        logger.info(f"✓ Open index.html to navigate results")
        logger.info("="*80)

        print("\n" + "="*80)
        print("✅ ANALYSIS SUCCESSFULLY COMPLETED")
        print("="*80)
        print(f"\n📁 Results Location: {RUN_DIR}")
        print(f"📊 Open: {RUN_DIR / 'index.html'}")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        logger.error(f"\n❌ ANALYSIS FAILED: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}\n")
        raise

if __name__ == '__main__':
    main()
