#!/usr/bin/env python
"""
Example Analysis Script for Agile Education Research
This script demonstrates practical usage of the analysis framework
with your Ukrainian VTT transcripts.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add framework to path if needed
sys.path.append('/mnt/user-data/outputs')

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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
TRANSCRIPTS_DIR = '/mnt/project'
OUTPUT_DIR = '/mnt/user-data/outputs'
RESULTS_DIR = Path(OUTPUT_DIR) / 'results'
VIZ_DIR = Path(OUTPUT_DIR) / 'visualizations'
REPORTS_DIR = Path(OUTPUT_DIR) / 'reports'

# Create output directories
for dir_path in [RESULTS_DIR, VIZ_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Load configuration
config = ResearchConfig()

# ============================================================================
# EXAMPLE 1: BASIC ANALYSIS PIPELINE
# ============================================================================

def run_basic_analysis():
    """Run the complete basic analysis pipeline"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Analysis Pipeline")
    print("=" * 60)
    
    # Initialize analyzer
    print("\n1. Initializing analyzer...")
    analyzer = AgileEducationAnalyzer(TRANSCRIPTS_DIR)
    
    # Run complete analysis
    print("2. Analyzing all sessions...")
    results = analyzer.analyze_all_sessions()
    
    # Save raw results
    print("3. Saving analysis results...")
    with open(RESULTS_DIR / 'complete_analysis.json', 'w', encoding='utf-8') as f:
        # Convert non-serializable objects
        serializable_results = {
            'engagement_evolution': results['engagement_evolution'].to_dict() if not results['engagement_evolution'].empty else {},
            'adoption_metrics': results['adoption_metrics'].to_dict() if not results['adoption_metrics'].empty else {},
            'problem_patterns': results['problem_patterns'].to_dict() if not results['problem_patterns'].empty else {},
            'recurring_problems': results['recurring_problems'],
            'topic_evolution': results['topic_evolution'].to_dict() if not results['topic_evolution'].empty else {},
            'statistical_results': results['statistical_results']
        }
        json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
    
    # Generate visualizations
    print("4. Generating visualizations...")
    analyzer.generate_visualizations(results, str(VIZ_DIR))
    
    # Generate report
    print("5. Generating research report...")
    analyzer.generate_report(results, str(REPORTS_DIR / 'analysis_report.md'))
    
    print("\n✅ Basic analysis complete!")
    return results

# ============================================================================
# EXAMPLE 2: FOCUSED SPRINT ANALYSIS
# ============================================================================

def analyze_sprint_progression():
    """Analyze progression across sprints"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Sprint Progression Analysis")
    print("=" * 60)
    
    processor = VTTProcessor()
    engagement_analyzer = EngagementAnalyzer()
    agile_analyzer = AgileAdoptionAnalyzer()
    
    sprint_data = {
        'sprint_1': [],
        'sprint_2': [],
        'sprint_3': []
    }
    
    # Process sprint files
    print("\n1. Processing sprint sessions...")
    for vtt_file in sorted(Path(TRANSCRIPTS_DIR).glob('*Спринт*.vtt')):
        print(f"   Processing: {vtt_file.name}")
        
        # Parse and analyze
        segments = processor.parse_vtt_file(str(vtt_file))
        metadata = processor.extract_session_metadata(str(vtt_file))
        
        # Identify speakers
        diarizer = SpeakerDiarization()
        segments = diarizer.identify_speakers(segments)
        
        # Calculate metrics
        engagement = engagement_analyzer.calculate_engagement_scores(segments)
        terminology = agile_analyzer.analyze_terminology_usage(segments)
        
        # Store by sprint
        if metadata.sprint_number:
            sprint_key = f'sprint_{metadata.sprint_number}'
            sprint_data[sprint_key].append({
                'file': vtt_file.name,
                'part': metadata.part_number,
                'engagement': engagement['overall'].mean() if not engagement.empty else 0,
                'questions': engagement['questions'].sum() if not engagement.empty else 0,
                'agile_terms': sum(t['frequency'] for t in terminology.values())
            })
    
    # Analyze progression
    print("\n2. Analyzing sprint progression...")
    progression_df = pd.DataFrame()
    
    for sprint, data in sprint_data.items():
        if data:
            sprint_summary = pd.DataFrame(data)
            sprint_summary['sprint'] = sprint
            progression_df = pd.concat([progression_df, sprint_summary])
    
    # Calculate statistics
    print("\n3. Sprint Statistics:")
    print("-" * 40)
    
    if not progression_df.empty:
        sprint_stats = progression_df.groupby('sprint').agg({
            'engagement': 'mean',
            'questions': 'sum',
            'agile_terms': 'sum'
        })
        
        print(sprint_stats.to_string())
        
        # Save results
        sprint_stats.to_csv(RESULTS_DIR / 'sprint_progression.csv')
        print(f"\n✅ Sprint analysis saved to {RESULTS_DIR / 'sprint_progression.csv'}")
    
    return progression_df

# ============================================================================
# EXAMPLE 3: STAND-UP EFFECTIVENESS ANALYSIS
# ============================================================================

def analyze_standup_effectiveness():
    """Analyze stand-up meeting effectiveness"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Stand-up Effectiveness Analysis")
    print("=" * 60)
    
    processor = VTTProcessor()
    problem_identifier = ProblemIdentifier()
    
    standup_results = []
    
    print("\n1. Processing stand-up sessions...")
    for vtt_file in sorted(Path(TRANSCRIPTS_DIR).glob('*Стендап*.vtt')):
        print(f"   Processing: {vtt_file.name}")
        
        # Parse file
        segments = processor.parse_vtt_file(str(vtt_file))
        metadata = processor.extract_session_metadata(str(vtt_file))
        
        # Calculate duration
        if segments:
            duration = (segments[-1].end_time - segments[0].start_time).total_seconds() / 60
        else:
            duration = 0
        
        # Identify problems discussed
        problems = problem_identifier.identify_problems(segments)
        
        # Count unique speakers (simplified)
        unique_speakers = len(set(s.text[:20] for s in segments))  # Simplified speaker count
        
        standup_results.append({
            'standup': vtt_file.name,
            'duration_minutes': duration,
            'segment_count': len(segments),
            'problems_discussed': len(problems),
            'estimated_speakers': unique_speakers,
            'avg_segment_length': sum(len(s.text.split()) for s in segments) / len(segments) if segments else 0
        })
    
    # Create DataFrame
    standup_df = pd.DataFrame(standup_results)
    
    print("\n2. Stand-up Meeting Statistics:")
    print("-" * 40)
    print(f"Average duration: {standup_df['duration_minutes'].mean():.1f} minutes")
    print(f"Average problems discussed: {standup_df['problems_discussed'].mean():.1f}")
    print(f"Average speakers: {standup_df['estimated_speakers'].mean():.1f}")
    
    # Identify trends
    print("\n3. Efficiency Trends:")
    if len(standup_df) > 1:
        # Check if stand-ups are getting shorter (more efficient)
        correlation = standup_df.index.to_series().corr(standup_df['duration_minutes'])
        if correlation < -0.3:
            print("✓ Stand-ups are becoming more efficient (shorter over time)")
        elif correlation > 0.3:
            print("⚠ Stand-ups are taking longer over time")
        else:
            print("→ Stand-up duration is relatively stable")
    
    # Save results
    standup_df.to_csv(RESULTS_DIR / 'standup_analysis.csv', index=False)
    print(f"\n✅ Stand-up analysis saved to {RESULTS_DIR / 'standup_analysis.csv'}")
    
    return standup_df

# ============================================================================
# EXAMPLE 4: QUALITATIVE CODING ANALYSIS
# ============================================================================

def run_qualitative_analysis():
    """Demonstrate qualitative coding analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Qualitative Coding Analysis")
    print("=" * 60)
    
    # Load all coding schemes
    print("\n1. Loading coding schemes...")
    all_codes = (
        EducationalCodingScheme.bloom_taxonomy_codes() +
        EducationalCodingScheme.interaction_codes() +
        EducationalCodingScheme.agile_learning_codes() +
        EducationalCodingScheme.problem_solving_codes()
    )
    
    coder = QualitativeCoder(all_codes)
    processor = VTTProcessor()
    
    # Process a sample file for demonstration
    sample_file = list(Path(TRANSCRIPTS_DIR).glob('*.vtt'))[0]
    print(f"\n2. Processing sample file: {sample_file.name}")
    
    segments = processor.parse_vtt_file(str(sample_file))[:20]  # First 20 segments
    
    # Code segments
    print("3. Applying qualitative codes...")
    coded_segments = []
    
    for segment in segments:
        codes = coder.auto_code(segment.text)
        coded_segment = {
            'text': segment.text[:100] + '...' if len(segment.text) > 100 else segment.text,
            'codes': codes,
            'timestamp': segment.start_time.total_seconds()
        }
        coded_segments.append(coded_segment)
    
    # Show sample results
    print("\n4. Sample Coding Results:")
    print("-" * 40)
    for i, seg in enumerate(coded_segments[:5]):
        print(f"\nSegment {i+1}:")
        print(f"Text: {seg['text']}")
        print(f"Codes: {', '.join(seg['codes']) if seg['codes'] else 'None'}")
    
    # Analyze code frequencies
    from collections import Counter
    code_freq = Counter()
    for seg in coded_segments:
        code_freq.update(seg['codes'])
    
    print("\n5. Code Frequency Analysis:")
    print("-" * 40)
    for code_id, count in code_freq.most_common(5):
        code = coder.codes.get(code_id)
        if code:
            print(f"{code.name}: {count} occurrences")
    
    # Save coding results
    coding_df = pd.DataFrame(coded_segments)
    coding_df.to_csv(RESULTS_DIR / 'qualitative_coding_sample.csv', index=False)
    print(f"\n✅ Coding results saved to {RESULTS_DIR / 'qualitative_coding_sample.csv'}")
    
    return coded_segments

# ============================================================================
# EXAMPLE 5: RESEARCH QUESTIONS ANALYSIS
# ============================================================================

def analyze_research_questions():
    """Analyze specific research questions"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Research Questions Analysis")
    print("=" * 60)
    
    # Run basic analysis first to get data
    analyzer = AgileEducationAnalyzer(TRANSCRIPTS_DIR)
    results = analyzer.analyze_all_sessions()
    
    print("\n" + "=" * 40)
    print("RESEARCH QUESTIONS ANSWERS")
    print("=" * 40)
    
    # RQ1: How does student participation evolve across sprints?
    print("\n📊 RQ1: Student Participation Evolution")
    print("-" * 40)
    
    evolution = results['engagement_evolution']
    if not evolution.empty and 'sprint_number' in evolution.columns:
        sprint_data = evolution[evolution['sprint_number'].notna()]
        if not sprint_data.empty:
            initial = sprint_data.iloc[0]['participation_rate']
            final = sprint_data.iloc[-1]['participation_rate']
            change = ((final - initial) / initial * 100) if initial > 0 else 0
            
            print(f"Initial participation rate: {initial:.1%}")
            print(f"Final participation rate: {final:.1%}")
            print(f"Change: {change:+.1f}%")
            
            if change > 10:
                print("→ Significant increase in participation")
            elif change < -10:
                print("→ Significant decrease in participation")
            else:
                print("→ Participation remained relatively stable")
    
    # RQ2: What agile concepts are most/least understood?
    print("\n📚 RQ2: Agile Concept Understanding")
    print("-" * 40)
    
    adoption = results['adoption_metrics']
    if not adoption.empty:
        avg_correct = adoption['correct_usage_rate'].mean()
        avg_misconception = adoption['misconception_rate'].mean()
        
        print(f"Average correct usage: {avg_correct:.1%}")
        print(f"Average misconception rate: {avg_misconception:.1%}")
        
        if avg_correct > 0.7:
            print("→ Good overall understanding of agile concepts")
        elif avg_correct > 0.5:
            print("→ Moderate understanding with room for improvement")
        else:
            print("→ Significant challenges in understanding agile concepts")
    
    # RQ3: What technical challenges emerge repeatedly?
    print("\n🔧 RQ3: Recurring Technical Challenges")
    print("-" * 40)
    
    recurring = results['recurring_problems']
    if recurring:
        print(f"Total recurring problems: {len(recurring)}")
        print("\nTop 3 recurring issues:")
        for i, problem in enumerate(recurring[:3], 1):
            print(f"{i}. {problem['category']}: {problem['occurrences']} times in {len(problem['sessions'])} sessions")
    else:
        print("No significant recurring problems identified")
    
    # RQ4: How effective are stand-ups for student learning?
    print("\n🎯 RQ4: Stand-up Effectiveness")
    print("-" * 40)
    
    standup_sessions = [s for s in results['sessions'] 
                        if s['metadata'].session_type == 'standup']
    regular_sessions = [s for s in results['sessions'] 
                       if s['metadata'].session_type != 'standup']
    
    if standup_sessions:
        standup_engagement = sum(s['engagement_scores']['overall'].mean() 
                                for s in standup_sessions if not s['engagement_scores'].empty) / len(standup_sessions)
        
        regular_engagement = sum(s['engagement_scores']['overall'].mean() 
                                for s in regular_sessions if not s['engagement_scores'].empty) / len(regular_sessions) if regular_sessions else 0
        
        print(f"Average stand-up engagement: {standup_engagement:.2f}")
        print(f"Average regular session engagement: {regular_engagement:.2f}")
        
        if standup_engagement > regular_engagement * 1.2:
            print("→ Stand-ups show significantly higher engagement")
        elif standup_engagement > regular_engagement:
            print("→ Stand-ups show slightly higher engagement")
        else:
            print("→ Stand-ups show similar or lower engagement")
    
    print("\n✅ Research questions analysis complete!")
    
    # Save summary
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_sessions': len(results['sessions']),
        'rq1_participation_change': change if 'change' in locals() else None,
        'rq2_correct_usage': avg_correct if 'avg_correct' in locals() else None,
        'rq3_recurring_problems': len(recurring) if recurring else 0,
        'rq4_standup_effectiveness': 'higher' if standup_sessions and standup_engagement > regular_engagement else 'similar'
    }
    
    with open(RESULTS_DIR / 'research_questions_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" AGILE EDUCATION RESEARCH ANALYSIS")
    print(" Example Analysis Pipeline")
    print("=" * 60)
    
    # Check if VTT files exist
    vtt_files = list(Path(TRANSCRIPTS_DIR).glob('*.vtt'))
    print(f"\nFound {len(vtt_files)} VTT transcript files")
    
    if not vtt_files:
        print("❌ No VTT files found in transcripts directory!")
        return
    
    # Run examples
    try:
        # Example 1: Basic analysis
        print("\n🔍 Running basic analysis...")
        basic_results = run_basic_analysis()
        
        # Example 2: Sprint progression
        print("\n📈 Analyzing sprint progression...")
        sprint_results = analyze_sprint_progression()
        
        # Example 3: Stand-up effectiveness
        print("\n⏱️ Analyzing stand-up meetings...")
        standup_results = analyze_standup_effectiveness()
        
        # Example 4: Qualitative coding
        print("\n📝 Running qualitative analysis...")
        coding_results = run_qualitative_analysis()
        
        # Example 5: Research questions
        print("\n❓ Analyzing research questions...")
        rq_summary = analyze_research_questions()
        
        # Final summary
        print("\n" + "=" * 60)
        print(" ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Results saved to: {RESULTS_DIR}")
        print(f"📊 Visualizations saved to: {VIZ_DIR}")
        print(f"📄 Reports saved to: {REPORTS_DIR}")
        
        print("\n🎯 Key Findings Summary:")
        print("-" * 40)
        if rq_summary:
            print(f"• Sessions analyzed: {rq_summary['total_sessions']}")
            if rq_summary['rq1_participation_change'] is not None:
                print(f"• Participation change: {rq_summary['rq1_participation_change']:+.1f}%")
            if rq_summary['rq2_correct_usage'] is not None:
                print(f"• Agile concept understanding: {rq_summary['rq2_correct_usage']:.1%}")
            print(f"• Recurring problems: {rq_summary['rq3_recurring_problems']}")
            print(f"• Stand-up effectiveness: {rq_summary['rq4_standup_effectiveness']}")
        
        print("\n✅ All analyses completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
