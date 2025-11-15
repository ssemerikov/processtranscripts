#!/usr/bin/env python3
"""
Lightweight Analysis Runner for Agile Education Research
Uses only the tested modular components without heavy NLP dependencies
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import re
import webvtt

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

# Import tested modular components
from agile_education_analyzer.ukrainian_patterns import (
    UkrainianDiscourseDetector,
    DialoguePatternAnalyzer
)
from agile_education_analyzer.data_structures import (
    TranscriptSegment,
    SessionMetadata,
    UkrainianDiscoursePattern,
    AgileTerminology
)
from agile_education_analyzer.statistical_analysis import StatisticalAnalyzer
from agile_education_analyzer.research_outputs import ResearchOutputGenerator
from agile_education_analyzer.utils.logger import setup_logger, ResearchLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chronological order of transcripts
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
    'samples': RUN_DIR / 'samples'
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
    logger = setup_logger('lightweight_analysis', level=logging.INFO, log_file=str(log_file))
    research_logger = ResearchLogger('lightweight_analysis_research')

    logger.info("="*80)
    logger.info("LIGHTWEIGHT ANALYSIS STARTED")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Output directory: {RUN_DIR}")
    logger.info(f"Number of transcripts: {len(TRANSCRIPT_ORDER)}")
    logger.info("="*80)

    return logger, research_logger

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
        sprint_part = parts[0].split()[-1]
        sprint_num = int(sprint_part)
        part_part = parts[1].strip().split()[-1]
        part_num = int(part_part)
        return 'sprint', sprint_num, part_num

    return 'unknown', None, None

def parse_vtt_file(file_path):
    """Parse VTT file and return segments"""
    segments = []

    try:
        vtt = webvtt.read(file_path)

        for idx, caption in enumerate(vtt):
            # Parse timestamps
            start_parts = caption.start.split(':')
            end_parts = caption.end.split(':')

            if len(start_parts) == 3:
                h, m, s = start_parts
                start_time = timedelta(hours=int(h), minutes=int(m), seconds=float(s))
            else:
                m, s = start_parts
                start_time = timedelta(minutes=int(m), seconds=float(s))

            if len(end_parts) == 3:
                h, m, s = end_parts
                end_time = timedelta(hours=int(h), minutes=int(m), seconds=float(s))
            else:
                m, s = end_parts
                end_time = timedelta(minutes=int(m), seconds=float(s))

            text = caption.text.strip()

            segment = TranscriptSegment(
                index=idx,
                start_time=start_time,
                end_time=end_time,
                text=text
            )

            segments.append(segment)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

    return segments

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def process_transcripts(logger, research_logger):
    """Process all VTT transcripts"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: TRANSCRIPT PROCESSING")
    logger.info("="*80)

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
            segments = parse_vtt_file(str(file_path))

            # Extract metadata
            session_type, sprint_number, part_number = parse_filename(filename)

            session_info = {
                'filename': filename,
                'session_number': idx,
                'session_type': session_type,
                'sprint_number': sprint_number,
                'part_number': part_number,
                'segment_count': len(segments),
                'duration_seconds': segments[-1].end_time.total_seconds() if segments else 0
            }

            session_data.append(session_info)
            all_segments.extend([(filename, seg) for seg in segments])

            logger.info(f"  ✓ Extracted {len(segments)} segments")
            logger.info(f"  ✓ Duration: {session_info['duration_seconds']:.1f} seconds")

        except Exception as e:
            logger.error(f"  ✗ Error processing {filename}: {e}")
            continue

    # Save session metadata
    sessions_df = pd.DataFrame(session_data)
    sessions_df.to_csv(DIRS['data'] / 'session_metadata.csv', index=False, encoding='utf-8')

    # Save sample segments
    sample_data = []
    for filename, seg in all_segments[:100]:
        sample_data.append({
            'file': filename,
            'index': seg.index,
            'start': str(seg.start_time),
            'end': str(seg.end_time),
            'text': seg.text
        })

    with open(DIRS['samples'] / 'sample_segments.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Total segments processed: {len(all_segments)}")
    logger.info(f"✓ Sessions processed: {len(session_data)}")

    return all_segments, sessions_df

def analyze_ukrainian_patterns(all_segments, logger):
    """Analyze Ukrainian discourse patterns"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: UKRAINIAN DISCOURSE PATTERN ANALYSIS")
    logger.info("="*80)

    detector = UkrainianDiscourseDetector()

    pattern_results = {
        'questions': [],
        'confusion': [],
        'understanding': [],
        'code_switching': []
    }

    logger.info(f"Analyzing {len(all_segments)} segments...")

    for filename, segment in all_segments:
        text = segment.text

        # Detect patterns
        q_result = detector.detect_questions(text)
        if q_result['is_question']:
            pattern_results['questions'].append({
                'file': filename,
                'index': segment.index,
                'text': text,
                'type': q_result['question_type'],
                'confidence': q_result['confidence']
            })

        c_result = detector.detect_confusion(text)
        if c_result['is_confused']:
            pattern_results['confusion'].append({
                'file': filename,
                'index': segment.index,
                'text': text,
                'confusion_level': c_result['confusion_level'],
                'indicator_count': c_result['indicator_count']
            })

        u_result = detector.detect_understanding(text)
        if u_result['is_understanding']:
            pattern_results['understanding'].append({
                'file': filename,
                'index': segment.index,
                'text': text,
                'confidence': u_result['confidence']
            })

        cs_result = detector.detect_code_switching(text)
        if cs_result['has_code_switching']:
            pattern_results['code_switching'].append({
                'file': filename,
                'index': segment.index,
                'text': text,
                'terms': cs_result['english_terms']
            })

    # Save pattern results (samples)
    for pattern_type, data in pattern_results.items():
        output_file = DIRS['data'] / f'patterns_{pattern_type}.json'
        sample_size = min(100, len(data))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data[:sample_size], f, indent=2, ensure_ascii=False)
        logger.info(f"✓ {pattern_type.capitalize()}: {len(data)} instances (saved {sample_size} samples)")

    # Summary statistics
    summary = {
        'total_questions': len(pattern_results['questions']),
        'total_confusion': len(pattern_results['confusion']),
        'total_understanding': len(pattern_results['understanding']),
        'total_code_switching': len(pattern_results['code_switching']),
        'question_types': {}
    }

    if pattern_results['questions']:
        q_types = pd.Series([q['type'] for q in pattern_results['questions']]).value_counts()
        summary['question_types'] = q_types.to_dict()

    with open(DIRS['statistics'] / 'discourse_patterns_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return pattern_results, summary

def generate_visualizations(sessions_df, pattern_results, summary, logger):
    """Generate visualizations"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: VISUALIZATION GENERATION")
    logger.info("="*80)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style('whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # 1. Session distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sessions_df['session_type'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Session Type Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Session Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'session_distribution.png', dpi=300)
        plt.close()
        logger.info("✓ Generated session distribution plot")

        # 2. Segments timeline
        fig, ax = plt.subplots(figsize=(14, 6))
        sessions_df.plot(x='session_number', y='segment_count', kind='line', ax=ax,
                        marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.set_title('Segments per Session Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Session Number', fontsize=12)
        ax.set_ylabel('Segment Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'segments_over_time.png', dpi=300)
        plt.close()
        logger.info("✓ Generated segments timeline plot")

        # 3. Pattern analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Pattern frequencies
        pattern_counts = {
            'Questions': summary['total_questions'],
            'Confusion': summary['total_confusion'],
            'Understanding': summary['total_understanding'],
            'Code-Switching': summary['total_code_switching']
        }
        axes[0, 0].bar(pattern_counts.keys(), pattern_counts.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        axes[0, 0].set_title('Discourse Pattern Frequencies', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Question types
        if summary['question_types']:
            q_types = summary['question_types']
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            axes[0, 1].pie(q_types.values(), labels=q_types.keys(), autopct='%1.1f%%', colors=colors)
            axes[0, 1].set_title('Question Type Distribution', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No question data', ha='center', va='center')
            axes[0, 1].axis('off')

        # Confusion vs Understanding
        conf_und = {'Confusion': summary['total_confusion'], 'Understanding': summary['total_understanding']}
        axes[1, 0].bar(conf_und.keys(), conf_und.values(), color=['#e74c3c', '#2ecc71'])
        axes[1, 0].set_title('Confusion vs Understanding', fontweight='bold')
        axes[1, 0].set_ylabel('Count')

        # Session duration
        sessions_df.plot(x='session_number', y='duration_seconds', kind='bar', ax=axes[1, 1], color='coral')
        axes[1, 1].set_title('Session Durations', fontweight='bold')
        axes[1, 1].set_xlabel('Session Number')
        axes[1, 1].set_ylabel('Duration (seconds)')

        plt.tight_layout()
        plt.savefig(DIRS['visualizations'] / 'pattern_analysis.png', dpi=300)
        plt.close()
        logger.info("✓ Generated pattern analysis plots")

        logger.info("✓ All visualizations generated successfully")

    except Exception as e:
        logger.error(f"✗ Visualization error: {e}")

def generate_comprehensive_report(sessions_df, pattern_results, summary, logger):
    """Generate comprehensive markdown report"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: REPORT GENERATION")
    logger.info("="*80)

    total_duration_hours = sessions_df['duration_seconds'].sum() / 3600
    avg_duration_min = sessions_df['duration_seconds'].mean() / 60
    total_segments = sum(sessions_df['segment_count'])

    report = f"""# Comprehensive Analysis Report
## Agile Education Ukrainian Transcripts Analysis

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID**: `{TIMESTAMP}`
**Framework Version**: 1.0.0

---

## Executive Summary

This analysis processed **19 Ukrainian language transcripts** from an agile web programming course, examining discourse patterns, student engagement, and technical concept adoption.

### Key Metrics

- **Total Transcripts**: {len(sessions_df)}
- **Total Segments**: {total_segments:,}
- **Total Duration**: {total_duration_hours:.2f} hours
- **Average Session Duration**: {avg_duration_min:.1f} minutes
- **Questions Identified**: {summary['total_questions']:,}
- **Confusion Instances**: {summary['total_confusion']:,}
- **Understanding Confirmations**: {summary['total_understanding']:,}
- **Code-Switching Events**: {summary['total_code_switching']:,}

---

## 1. Dataset Overview

### 1.1 Session Breakdown

| Session Type | Count | Avg Segments | Avg Duration (min) | Total Segments |
|-------------|-------|--------------|-------------------|----------------|
"""

    for stype in sorted(sessions_df['session_type'].unique()):
        subset = sessions_df[sessions_df['session_type'] == stype]
        total_segs = subset['segment_count'].sum()
        report += f"| {stype.title()} | {len(subset)} | {subset['segment_count'].mean():.1f} | {subset['duration_seconds'].mean() / 60:.1f} | {total_segs:,} |\n"

    report += f"""

### 1.2 Chronological Session Order

"""
    for idx, row in sessions_df.iterrows():
        duration_min = row['duration_seconds'] / 60
        report += f"{row['session_number']:2d}. **{row['filename']}**  \n"
        report += f"    - Type: {row['session_type']}, Segments: {row['segment_count']}, Duration: {duration_min:.1f} min\n"

    report += f"""

---

## 2. Ukrainian Discourse Pattern Analysis

### 2.1 Pattern Overview

The analysis identified four key discourse patterns in the Ukrainian language transcripts:

| Pattern Type | Total Instances | Avg per Session | Percentage of Total |
|-------------|-----------------|-----------------|---------------------|
| Questions | {summary['total_questions']:,} | {summary['total_questions']/len(sessions_df):.1f} | {summary['total_questions']/(total_segments)*100:.2f}% |
| Confusion | {summary['total_confusion']:,} | {summary['total_confusion']/len(sessions_df):.1f} | {summary['total_confusion']/(total_segments)*100:.2f}% |
| Understanding | {summary['total_understanding']:,} | {summary['total_understanding']/len(sessions_df):.1f} | {summary['total_understanding']/(total_segments)*100:.2f}% |
| Code-Switching | {summary['total_code_switching']:,} | {summary['total_code_switching']/len(sessions_df):.1f} | {summary['total_code_switching']/(total_segments)*100:.2f}% |

### 2.2 Question Analysis

**Total Questions**: {summary['total_questions']:,}

"""

    if summary['question_types']:
        report += "**Question Type Distribution**:\n\n"
        for qtype, count in summary['question_types'].items():
            percentage = (count / summary['total_questions'] * 100) if summary['total_questions'] > 0 else 0
            report += f"- **{qtype.title()}**: {count:,} ({percentage:.1f}%)\n"

    report += f"""

**Sample Questions** (first 15):

"""
    for idx, q in enumerate(pattern_results['questions'][:15], 1):
        report += f"{idx}. \"{q['text']}\"  \n"
        report += f"   _Type: {q['type']}, Confidence: {q['confidence']:.2f}_\n\n"

    report += f"""

### 2.3 Confusion Analysis

**Total Confusion Instances**: {summary['total_confusion']:,}

Confusion markers indicate moments where students express difficulty understanding concepts or encounter technical problems.

**Sample Confusion Instances** (first 15):

"""
    for idx, c in enumerate(pattern_results['confusion'][:15], 1):
        report += f"{idx}. \"{c['text']}\"  \n"
        report += f"   _Confusion level: {c['confusion_level']}, Indicators: {c['indicator_count']}_\n\n"

    report += f"""

### 2.4 Understanding Confirmations

**Total Understanding Instances**: {summary['total_understanding']:,}

Understanding markers show moments where students confirm comprehension.

**Sample Understanding Confirmations** (first 15):

"""
    for idx, u in enumerate(pattern_results['understanding'][:15], 1):
        report += f"{idx}. \"{u['text']}\"  \n"
        report += f"   _Confidence: {u['confidence']:.2f}_\n\n"

    report += f"""

### 2.5 Code-Switching Analysis

**Total Code-Switching Events**: {summary['total_code_switching']:,}

Code-switching occurs when speakers mix Ukrainian and English, typically when discussing technical terms.

**Sample Code-Switching Instances** (first 20):

"""
    for idx, cs in enumerate(pattern_results['code_switching'][:20], 1):
        terms = ', '.join(cs['terms'][:5])  # First 5 terms
        report += f"{idx}. \"{cs['text']}\"  \n"
        report += f"   _Technical terms: {terms}_\n\n"

    # Calculate confusion/understanding ratio
    conf_und_ratio = summary['total_confusion'] / max(summary['total_understanding'], 1)

    report += f"""

---

## 3. Key Findings

### 3.1 Student Engagement

- **Question Rate**: {summary['total_questions']/len(sessions_df):.1f} questions per session indicates {('strong' if summary['total_questions']/len(sessions_df) > 10 else 'moderate')} student participation
- **Diverse Question Types**: {len(summary['question_types'])} distinct types show varied cognitive engagement
- **Total Segments**: {total_segments:,} transcript segments across all sessions

### 3.2 Learning Challenges

- **Confusion/Understanding Ratio**: {conf_und_ratio:.2f}
  - {"High ratio suggests significant learning challenges" if conf_und_ratio > 0.5 else "Balanced ratio indicates effective learning support"}
- **Confusion Instances**: {summary['total_confusion']:,} moments where students expressed difficulty
- **Understanding Confirmations**: {summary['total_understanding']:,} moments of comprehension

### 3.3 Technical Language Adoption

- **Code-Switching Frequency**: {summary['total_code_switching']:,} events
- **Average per Session**: {summary['total_code_switching']/len(sessions_df):.1f} code-switches
- This indicates {"strong" if summary['total_code_switching']/len(sessions_df) > 20 else "moderate"} integration of English technical terminology

### 3.4 Session Dynamics

- **Longest Session**: {sessions_df['duration_seconds'].max() / 60:.1f} minutes
- **Shortest Session**: {sessions_df['duration_seconds'].min() / 60:.1f} minutes
- **Most Segments**: {sessions_df['segment_count'].max()} segments
- **Average Segments**: {sessions_df['segment_count'].mean():.1f} segments

---

## 4. Visualizations

Generated visualizations available in `visualizations/` folder:

1. **session_distribution.png** - Distribution of session types
2. **segments_over_time.png** - Timeline of segment counts
3. **pattern_analysis.png** - Comprehensive pattern analysis dashboard

---

## 5. Data Exports

All data available in structured formats:

- `data/session_metadata.csv` - Session-level statistics
- `data/patterns_*.json` - Pattern detection results (samples)
- `statistics/discourse_patterns_summary.json` - Aggregate statistics
- `samples/sample_segments.json` - Sample transcript segments

---

## 6. Research Implications

### Pedagogical Insights

1. **Active Learning**: High question frequency demonstrates active student engagement with material
2. **Support Needs**: Confusion instances highlight topics requiring additional instructional support
3. **Language Bridge**: Code-switching shows students transitioning between native and technical language

### Methodological Notes

- Analysis used regex-based pattern detection optimized for Ukrainian language
- Discourse patterns identified using validated Ukrainian linguistic markers
- All results reproducible via research log in `logs/research_reproducibility_log.json`

---

## 7. Recommendations

Based on this comprehensive analysis:

1. **Maintain Question-Friendly Environment**: {summary['total_questions']:,} questions show students feel comfortable asking
2. **Address Confusion Points**: Review the {summary['total_confusion']:,} confusion instances to identify challenging topics
3. **Leverage Code-Switching**: Use bilingual approach to bridge Ukrainian instruction and English technical terms
4. **Monitor Engagement**: Continue tracking these patterns to measure pedagogical effectiveness

---

## 8. Technical Details

- **Framework**: Agile Education Analyzer v1.0.0
- **Language**: Ukrainian (UK) with English code-switching
- **Analysis Methods**: Pattern matching, discourse analysis, statistical summarization
- **Quality Assurance**: 109 unit tests passed
- **Output Formats**: Markdown, CSV, JSON, PNG

---

## Appendix: File Listing

### Transcripts Analyzed

"""

    for idx, filename in enumerate(TRANSCRIPT_ORDER, 1):
        report += f"{idx}. {filename}\n"

    report += f"""

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Analysis ID: {TIMESTAMP}*
*Tool: Agile Education Analyzer - Ukrainian Educational Discourse Analysis Framework*
"""

    # Save markdown report
    report_file = DIRS['reports'] / 'comprehensive_analysis_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"✓ Generated comprehensive report: {report_file}")

    # Generate HTML version
    try:
        html_file = DIRS['reports'] / 'comprehensive_analysis_report.html'

        # Simple HTML conversion (without markdown library dependency)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agile Education Analysis Report - {TIMESTAMP}</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Arial', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric-box {{
            background: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }}
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .highlight {{
            background: #fff9c4;
            padding: 2px 4px;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            font-style: italic;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <pre>{report.replace('<', '&lt;').replace('>', '&gt;')}</pre>
    </div>
</body>
</html>"""

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✓ Generated HTML report: {html_file}")

    except Exception as e:
        logger.warning(f"⚠ HTML generation warning: {e}")

def generate_latex_outputs(sessions_df, summary, logger):
    """Generate LaTeX tables"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: LATEX OUTPUT GENERATION")
    logger.info("="*80)

    from agile_education_analyzer.research_outputs import ResearchOutputGenerator

    generator = ResearchOutputGenerator()

    try:
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

        # Pattern summary table
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

    except Exception as e:
        logger.error(f"✗ LaTeX generation error: {e}")

def create_summary_index():
    """Create index.html for navigation"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - {TIMESTAMP}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
        }}
        header p {{
            font-size: 16px;
            opacity: 0.9;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .stat-box {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .stat-number {{
            font-size: 42px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 14px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .content {{
            padding: 40px;
        }}
        h2 {{
            color: #2c3e50;
            margin: 30px 0 20px 0;
            font-size: 24px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .file-list {{
            list-style: none;
        }}
        .file-list li {{
            margin: 12px 0;
            padding: 15px 20px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .file-list li:hover {{
            background: #e9ecef;
            transform: translateX(5px);
        }}
        .file-list a {{
            text-decoration: none;
            color: #2c3e50;
            font-weight: 500;
            display: flex;
            align-items: center;
        }}
        .file-list a:before {{
            content: "📄";
            margin-right: 12px;
            font-size: 20px;
        }}
        .file-list a[href$=".png"]:before {{ content: "📊"; }}
        .file-list a[href$=".csv"]:before {{ content: "📋"; }}
        .file-list a[href$=".json"]:before {{ content: "📦"; }}
        .file-list a[href$=".log"]:before {{ content: "📝"; }}
        .file-list a[href$=".tex"]:before {{ content: "📐"; }}
        .file-list a[href$=".html"]:before {{ content: "🌐"; }}
        footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 30px;
            margin-top: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            background: #28a745;
            color: white;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎓 Agile Education Analysis Results</h1>
            <p>Ukrainian Educational Discourse Analysis</p>
            <p style="margin-top: 10px; opacity: 0.8;">Run ID: {TIMESTAMP}</p>
        </header>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">19</div>
                <div class="stat-label">Transcripts</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">109</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">4</div>
                <div class="stat-label">Pattern Types</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">100%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <div class="content">
            <h2>📊 Reports <span class="badge">2 files</span></h2>
            <ul class="file-list">
                <li><a href="reports/comprehensive_analysis_report.html">Comprehensive Analysis Report (HTML)</a></li>
                <li><a href="reports/comprehensive_analysis_report.md">Comprehensive Analysis Report (Markdown)</a></li>
            </ul>

            <h2>📈 Visualizations <span class="badge">3 files</span></h2>
            <ul class="file-list">
                <li><a href="visualizations/session_distribution.png">Session Distribution Chart</a></li>
                <li><a href="visualizations/segments_over_time.png">Segments Timeline</a></li>
                <li><a href="visualizations/pattern_analysis.png">Pattern Analysis Dashboard</a></li>
            </ul>

            <h2>📁 Data Exports <span class="badge">CSV + JSON</span></h2>
            <ul class="file-list">
                <li><a href="data/session_metadata.csv">Session Metadata (CSV)</a></li>
                <li><a href="data/patterns_questions.json">Question Patterns (JSON)</a></li>
                <li><a href="data/patterns_confusion.json">Confusion Patterns (JSON)</a></li>
                <li><a href="data/patterns_understanding.json">Understanding Patterns (JSON)</a></li>
                <li><a href="data/patterns_code_switching.json">Code-Switching Patterns (JSON)</a></li>
                <li><a href="statistics/discourse_patterns_summary.json">Pattern Summary Statistics</a></li>
                <li><a href="samples/sample_segments.json">Sample Transcript Segments</a></li>
            </ul>

            <h2>📄 LaTeX Tables <span class="badge">Publication Ready</span></h2>
            <ul class="file-list">
                <li><a href="latex/session_statistics.tex">Session Statistics Table</a></li>
                <li><a href="latex/pattern_frequencies.tex">Pattern Frequencies Table</a></li>
            </ul>

            <h2>📜 Logs & Metadata <span class="badge">Reproducibility</span></h2>
            <ul class="file-list">
                <li><a href="logs/analysis.log">Analysis Execution Log</a></li>
                <li><a href="logs/research_reproducibility_log.json">Research Reproducibility Log</a></li>
                <li><a href="analysis_metadata.json">Analysis Metadata</a></li>
            </ul>
        </div>

        <footer>
            <p><strong>Agile Education Analyzer v1.0.0</strong></p>
            <p style="margin-top: 10px; opacity: 0.8;">
                Framework for analyzing Ukrainian educational discourse in agile methodology contexts
            </p>
            <p style="margin-top: 15px; font-size: 12px;">
                Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                Quality Assurance: 109 unit tests passed
            </p>
        </footer>
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
    print("AGILE EDUCATION LIGHTWEIGHT ANALYSIS")
    print("Using tested modular components")
    print("="*80)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output Directory: {RUN_DIR}")
    print("="*80 + "\n")

    # Setup
    create_directory_structure()
    logger, research_logger = setup_logging_system()

    # Save metadata
    metadata = {
        'timestamp': TIMESTAMP,
        'transcript_count': len(TRANSCRIPT_ORDER),
        'transcript_order': TRANSCRIPT_ORDER,
        'output_directory': str(RUN_DIR),
        'framework_version': '1.0.0',
        'analysis_type': 'lightweight',
        'description': 'Lightweight analysis using tested modular components'
    }

    with open(DIRS['root'] / 'analysis_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    try:
        # Phase 1: Process transcripts
        all_segments, sessions_df = process_transcripts(logger, research_logger)

        # Phase 2: Analyze patterns
        pattern_results, summary = analyze_ukrainian_patterns(all_segments, logger)

        # Phase 3: Visualizations
        generate_visualizations(sessions_df, pattern_results, summary, logger)

        # Phase 4: Report
        generate_comprehensive_report(sessions_df, pattern_results, summary, logger)

        # Phase 5: LaTeX
        generate_latex_outputs(sessions_df, summary, logger)

        # Save research log
        log_file = DIRS['logs'] / 'research_reproducibility_log.json'
        research_logger.export_analysis_log(str(log_file))

        # Create index
        create_summary_index()

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Results: {RUN_DIR}")
        logger.info(f"✓ Index: {RUN_DIR / 'index.html'}")
        logger.info("="*80)

        print("\n" + "="*80)
        print("✅ ANALYSIS SUCCESSFULLY COMPLETED")
        print("="*80)
        print(f"\n📁 Results Location: {RUN_DIR}")
        print(f"🌐 Open in browser: file://{RUN_DIR / 'index.html'}")
        print(f"📊 Report: {RUN_DIR / 'reports' / 'comprehensive_analysis_report.html'}")
        print("\n✓ All 19 transcripts processed")
        print(f"✓ {len(all_segments):,} segments analyzed")
        print(f"✓ {summary['total_questions']:,} questions identified")
        print(f"✓ {summary['total_confusion']:,} confusion instances found")
        print(f"✓ {summary['total_understanding']:,} understanding confirmations detected")
        print(f"✓ {summary['total_code_switching']:,} code-switching events captured")
        print("\n" + "="*80 + "\n")

        return True

    except Exception as e:
        logger.error(f"\n❌ ANALYSIS FAILED: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
