# Agile Education Analysis Framework - Usage Guide

## Overview

This comprehensive framework is designed for analyzing Ukrainian VTT transcripts from educational sessions on agile web programming methodologies. It provides tools for both quantitative and qualitative analysis, following established educational research methodologies.

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt

# Download Ukrainian language models
python -m spacy download uk_core_news_sm
python -m stanza.download uk

# For advanced NLP (optional)
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='youscan/ukr-roberta-sentiment')"
```

### 2. Basic Usage

```python
from agile_education_analysis_framework import AgileEducationAnalyzer

# Initialize analyzer with your transcripts directory
analyzer = AgileEducationAnalyzer('/mnt/project')

# Run complete analysis
results = analyzer.analyze_all_sessions()

# Generate visualizations
analyzer.generate_visualizations(results, './output/visualizations')

# Generate research report
analyzer.generate_report(results, './output/analysis_report.md')
```

## Framework Components

### 1. Data Processing Strategy

#### VTT File Parsing
- **Purpose**: Extract structured data from WebVTT subtitle files
- **Features**:
  - Handles Ukrainian text with proper Unicode normalization
  - Removes transcription service watermarks
  - Preserves timestamps for temporal analysis
  - Fixes common OCR/transcription errors

```python
from agile_education_analysis_framework import VTTProcessor

processor = VTTProcessor()
segments = processor.parse_vtt_file('path/to/file.vtt')
metadata = processor.extract_session_metadata('path/to/file.vtt')
```

#### Speaker Diarization
- **Purpose**: Identify and differentiate speakers (teachers vs. students)
- **Methods**:
  - Pattern-based identification using linguistic markers
  - Clustering using multilingual sentence embeddings
  - Contextual role assignment

```python
from agile_education_analysis_framework import SpeakerDiarization

diarizer = SpeakerDiarization()
segments_with_speakers = diarizer.identify_speakers(segments)
```

### 2. Key Metrics & Patterns

#### Student Engagement Indicators
- **Question frequency**: Tracks how often students ask questions
- **Active participation**: Measures proactive contributions
- **Confusion indicators**: Identifies when students struggle
- **Understanding markers**: Detects comprehension signals
- **Technical discussions**: Monitors depth of technical engagement

```python
from agile_education_analysis_framework import EngagementAnalyzer

analyzer = EngagementAnalyzer()
engagement_scores = analyzer.calculate_engagement_scores(segments)
participation_stats = analyzer.analyze_participation_frequency(segments)
evolution = analyzer.track_engagement_evolution(sessions)
```

#### Agile Adoption Patterns
- **Terminology usage**: Tracks agile vocabulary adoption
- **Correct vs. incorrect usage**: Identifies misconceptions
- **Student adoption rate**: Measures how students incorporate agile terms
- **Concept progression**: Monitors learning curve across sprints

```python
from agile_education_analysis_framework import AgileAdoptionAnalyzer

agile_analyzer = AgileAdoptionAnalyzer()
terminology_usage = agile_analyzer.analyze_terminology_usage(segments)
adoption_metrics = agile_analyzer.calculate_adoption_metrics(sessions)
```

#### Problem Identification
- **Technical issues**: Code errors, environment problems
- **Conceptual difficulties**: Understanding challenges
- **Process challenges**: Time management, workflow issues
- **Collaboration issues**: Team dynamics problems
- **Tool problems**: Software/setup issues

```python
from agile_education_analysis_framework import ProblemIdentifier

identifier = ProblemIdentifier()
problems = identifier.identify_problems(segments)
patterns = identifier.analyze_problem_patterns(sessions)
recurring = identifier.find_recurring_problems(sessions, similarity_threshold=0.7)
```

### 3. Analysis Methodologies

#### Qualitative Coding
The framework includes a comprehensive qualitative coding system with:
- **Bloom's Taxonomy codes**: Cognitive levels (Remember to Create)
- **Interaction codes**: Teacher-student communication patterns
- **Agile learning codes**: Methodology-specific indicators
- **Problem-solving codes**: Solution development process

```python
from qualitative_coding import QualitativeCoder, EducationalCodingScheme

# Initialize with educational coding schemes
codes = EducationalCodingScheme.bloom_taxonomy_codes()
coder = QualitativeCoder(codes)

# Auto-code segments
coded_segments = []
for segment in segments:
    codes = coder.auto_code(segment.text, segment.speaker)
    coded_segment = CodedSegment(text=segment.text, codes=codes)
    coded_segments.append(coded_segment)

# Validate coding
validation = coder.validate_coding(coded_segments)
```

#### Sentiment Analysis for Ukrainian
- Uses Ukrainian-specific models when available
- Falls back to multilingual models if needed
- Tracks emotional tone throughout sessions

```python
from agile_education_analysis_framework import SentimentTopicAnalyzer

sentiment_analyzer = SentimentTopicAnalyzer()
sentiment_scores = sentiment_analyzer.analyze_sentiment(segments)
```

#### Topic Modeling
- Latent Dirichlet Allocation (LDA) for topic extraction
- Ukrainian stop words filtering
- N-gram support for phrase detection
- Topic evolution tracking across sessions

```python
topics = sentiment_analyzer.extract_topics(segments, n_topics=10)
topic_evolution = sentiment_analyzer.analyze_topic_evolution(sessions)
```

### 4. Statistical Analysis

#### Comparative Tests
- **Kruskal-Wallis**: Compare metrics across multiple sprints
- **Mann-Whitney U**: Pairwise sprint comparisons
- **Effect sizes**: Cohen's d for practical significance
- **Bonferroni correction**: Multiple comparison adjustment

```python
from agile_education_analysis_framework import StatisticalAnalyzer

stats_analyzer = StatisticalAnalyzer()
comparison = stats_analyzer.compare_sprints(sprint_data, 'participation_rate')
correlation = stats_analyzer.analyze_correlation(data, 'var1', 'var2')
trend = stats_analyzer.time_series_analysis(data, 'metric')
```

### 5. Visualization & Reporting

#### Key Visualizations
1. **Engagement Evolution**: Line plots showing metrics across sprints
2. **Agile Adoption**: Dual plots for adoption rates and terminology usage
3. **Problem Heatmaps**: Session × problem category matrices
4. **Speaker Networks**: Interaction flow diagrams
5. **Word Clouds**: Sprint-specific vocabulary visualization

```python
from agile_education_analysis_framework import ResearchVisualizer

visualizer = ResearchVisualizer()
visualizer.plot_engagement_evolution(evolution_data, 'output.png')
visualizer.plot_agile_adoption(adoption_data, 'adoption.png')
visualizer.create_problem_heatmap(problem_patterns, 'heatmap.png')
visualizer.create_speaker_network(interactions, 'network.png')
visualizer.create_wordcloud_by_sprint(sessions, 'wordclouds.png')
```

## Research Questions Framework

### RQ1: Student Participation Evolution
**Metrics to analyze**:
- `participation_rate`: Percentage of active contributors
- `question_frequency`: Questions per student per session
- `utterance_count`: Total student contributions

**Analysis approach**:
```python
# Track participation across sprints
evolution = engagement_analyzer.track_engagement_evolution(sessions)

# Statistical comparison
stats = statistical_analyzer.compare_sprints(evolution, 'participation_rate')

# Visualize trend
visualizer.plot_engagement_evolution(evolution, 'participation_trend.png')
```

### RQ2: Agile Concept Understanding
**Metrics to analyze**:
- `correct_usage_rate`: Proper terminology application
- `misconception_rate`: Incorrect concept usage
- `term_frequency`: Vocabulary adoption rate

**Analysis approach**:
```python
# Analyze terminology patterns
for session in sessions:
    usage = agile_analyzer.analyze_terminology_usage(session['segments'])
    
# Calculate adoption metrics
adoption = agile_analyzer.calculate_adoption_metrics(sessions)

# Identify common misconceptions
misconceptions = [u for u in usage.values() if u['misconceptions'] > 0]
```

### RQ3: Technical Challenges
**Metrics to analyze**:
- `problem_frequency`: Issue occurrence rate
- `problem_categories`: Distribution of problem types
- `resolution_time`: Time to solve problems

**Analysis approach**:
```python
# Identify all problems
all_problems = []
for session in sessions:
    problems = problem_identifier.identify_problems(session['segments'])
    all_problems.extend(problems)

# Find patterns
patterns = problem_identifier.analyze_problem_patterns(sessions)
recurring = problem_identifier.find_recurring_problems(sessions)

# Categorize by sprint
sprint_problems = patterns.groupby('sprint_number')['total_problems'].sum()
```

### RQ4: Stand-up Effectiveness
**Metrics to analyze**:
- `standup_engagement`: Participation in stand-ups
- `problem_reporting`: Issues raised in stand-ups
- `peer_interaction`: Student-to-student communication

**Analysis approach**:
```python
# Filter stand-up sessions
standups = [s for s in sessions if s['metadata'].session_type == 'standup']

# Calculate stand-up specific metrics
standup_metrics = []
for standup in standups:
    metrics = {
        'engagement': standup['engagement_scores']['overall'].mean(),
        'problems_reported': len(standup['problems']),
        'peer_interactions': sum(1 for i in standup['interaction_patterns'] 
                               if 'peer' in i)
    }
    standup_metrics.append(metrics)

# Compare with regular sessions
comparison = stats.mannwhitneyu(standup_engagement, regular_engagement)
```

## Advanced Features

### Inter-Rater Reliability
For validating qualitative coding consistency:

```python
from qualitative_coding import InterRaterReliability

# Compare two coders' results
reliability = InterRaterReliability.calculate_agreement(
    coder1_segments, 
    coder2_segments,
    code_list
)

print(f"Cohen's Kappa: {reliability['overall']['avg_cohen_kappa']:.3f}")
```

### Code Co-occurrence Analysis
Identify patterns in code relationships:

```python
from qualitative_coding import CodeCoOccurrenceAnalyzer

analyzer = CodeCoOccurrenceAnalyzer(coded_segments)
co_occurrence = analyzer.calculate_co_occurrence()
clusters = analyzer.find_code_clusters(threshold=0.5)
temporal = analyzer.analyze_temporal_patterns()
```

### Thematic Analysis
Extract emergent themes from coded data:

```python
from qualitative_coding import ThematicAnalyzer

thematic = ThematicAnalyzer(coded_segments, codes)
themes = thematic.identify_themes()
hierarchy = thematic.create_theme_hierarchy()
```

## Configuration

Customize analysis parameters via `research_config.py`:

```python
from research_config import ResearchConfig

config = ResearchConfig()

# Modify parameters
config.set('analysis.topic_modeling.n_topics', 15)
config.set('statistics.significance_level', 0.01)

# Access configuration
n_topics = config.get('analysis.topic_modeling.n_topics')
```

## Output Formats

### Research Report (Markdown)
Automatically generated comprehensive report including:
- Executive summary
- Key findings by research question
- Statistical results
- Visualizations
- Recommendations

### Data Exports
- **CSV**: Tabular data for further analysis
- **JSON**: Structured data with metadata
- **Excel**: Multi-sheet workbook with results
- **LaTeX**: Tables formatted for academic papers

```python
# Export to different formats
df = reporter.export_to_dataframe()
df.to_csv('results.csv', encoding='utf-8')
df.to_excel('results.xlsx', index=False)

# Statistical export
stats_data = reporter.export_for_statistical_analysis()
with open('stats_data.json', 'w') as f:
    json.dump(stats_data, f, ensure_ascii=False)
```

## Best Practices

### 1. Data Preparation
- Ensure VTT files are properly encoded (UTF-8)
- Verify Ukrainian text is correctly displayed
- Check timestamp continuity

### 2. Analysis Pipeline
- Start with data validation
- Run speaker diarization before other analyses
- Perform qualitative coding early for context
- Save intermediate results for reproducibility

### 3. Interpretation
- Consider cultural context in Ukrainian education
- Account for technical terminology mixing (Ukrainian/English)
- Validate findings with multiple analysis methods
- Use triangulation for robust conclusions

### 4. Reporting
- Follow APA guidelines for educational research
- Include effect sizes with statistical tests
- Provide confidence intervals
- Discuss limitations and threats to validity

## Troubleshooting

### Common Issues

1. **Ukrainian text not displaying correctly**
   - Check file encoding: should be UTF-8
   - Verify font support for Cyrillic characters

2. **Speaker diarization accuracy low**
   - Adjust clustering parameters in config
   - Add more linguistic patterns for your context
   - Consider manual validation sample

3. **Topic modeling producing unclear topics**
   - Increase/decrease number of topics
   - Expand stop words list
   - Try different n-gram ranges

4. **Memory issues with large datasets**
   - Process sessions in batches
   - Use generators instead of lists
   - Reduce embedding model size

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{agile_education_analyzer,
  title = {Agile Education Analysis Framework for Ukrainian Transcripts},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/agile-education-analysis}
}
```

## Support

For questions or issues:
1. Check this documentation
2. Review example notebooks in `/examples`
3. Consult the API reference
4. Submit an issue on GitHub

## License

This framework is provided for educational research purposes. Please ensure you have appropriate permissions for transcript analysis and comply with relevant privacy regulations.
