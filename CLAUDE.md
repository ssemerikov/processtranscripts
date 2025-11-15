# CLAUDE.md - AI Assistant Guide for ProcessTranscripts Repository

## Repository Identity & Purpose

This repository represents a **specialized educational research framework** for analyzing Ukrainian-language transcripts from an agile web programming course. It is not a general-purpose tool, but rather a purpose-built research instrument combining:

- **Educational Research Methodology**: Mixed-methods approach with rigorous qualitative and quantitative analysis
- **Ukrainian Natural Language Processing**: Sophisticated handling of Cyrillic text, morphology, and linguistic patterns
- **Agile Methodology Domain Knowledge**: Deep understanding of Scrum, sprints, stand-ups, and agile pedagogy
- **Data Science**: Topic modeling, sentiment analysis, statistical testing, and visualization

**Core Research Question**: How effectively do agile methodologies translate to educational contexts, specifically for teaching web programming to Ukrainian-speaking students?

## Critical Context for AI Assistants

### 1. Language Complexity

**Ukrainian Language Considerations**:
- **Cyrillic alphabet**: All transcripts use Ukrainian Cyrillic characters (UTF-8 encoding essential)
- **Rich morphology**: Ukrainian has 7 cases, 3 genders, complex verb conjugations
- **Code-switching**: Students and teachers mix Ukrainian and English (especially for technical terms)
- **Informal speech**: Transcripts capture live classroom discussion, not formal writing
- **NLP challenges**:
  - Limited Ukrainian language models compared to English
  - Need for specialized libraries (pymorphy3, Stanza, spaCy)
  - Sentiment analysis models are scarce for Ukrainian
  - Technical terminology often borrowed from English

**Key Insight**: Do not assume English NLP approaches will work. Ukrainian requires specialized processing pipelines, and fallback strategies are essential when Ukrainian-specific models are unavailable.

### 2. Educational Research Domain

**This is NOT software analytics** - it's educational research with specific methodological requirements:

- **Bloom's Taxonomy**: Cognitive levels from remembering to creating must be recognized in student discourse
- **Qualitative Coding**: Systematic coding schemes applied to transcript segments
- **Inter-Rater Reliability**: Multiple coders must achieve agreement (Cohen's Kappa)
- **Triangulation**: Multiple data sources/methods validate findings
- **Ethical Considerations**: Student privacy, anonymization, informed consent
- **Pedagogical Theory**: Understanding of constructivist learning, active learning, assessment

**Research Rigor Requirements**:
- Statistical tests must be appropriate for non-normal distributions (hence Kruskal-Wallis, Mann-Whitney)
- Effect sizes matter as much as p-values (Cohen's d)
- Multiple comparison corrections (Bonferroni) are necessary
- Qualitative findings require thick description and member checking

### 3. Agile Methodology Context

**Agile in Education is Different**:
- Students are learning agile while using agile (meta-learning)
- "Sprints" are educational units, not software development cycles
- "Stand-ups" are pedagogical tools for reflection and peer learning
- Agile terminology adoption is itself a learning outcome being measured

**Key Agile Concepts Tracked**:
- Sprint planning, execution, review, retrospective
- Stand-up meetings (blockers, progress, plans)
- User stories, backlog management
- Velocity, burndown, team dynamics
- Technical practices (CI/CD, testing, version control)

### 4. Data Structure & Format

**VTT (WebVTT) Transcripts**:
- **Format**: Plain text with timestamps and optional speaker labels
- **Structure**:
  ```
  WEBVTT

  00:00:05.000 --> 00:00:08.000
  Сьогодні ми почнемо вивчати Scrum методологію

  00:00:08.500 --> 00:00:12.000
  Хто знає, що таке спринт?
  ```
- **Challenges**:
  - Speaker labels may be inconsistent or missing
  - Transcription errors from automatic speech recognition
  - Timestamps may have gaps or overlaps
  - Watermarks from transcription services need removal

**19 Transcript Files** (in chronological order):
1. Web2.П01.Вступ до гнучкої розробки.vtt (Introduction to Agile Development)
2. Web2.Стендап 1.vtt (Standup 1)
3. Web2.П02.Спринт 1, частина 1.vtt (Sprint 1, Part 1)
4. Web2.Стендап 2.vtt (Standup 2)
5. Web2.П03.Спринт 1, частина 2.vtt (Sprint 1, Part 2)
6. Web2.Стендап 3.vtt (Standup 3)
7. Web2.П04.Спринт 1, частина 3.vtt (Sprint 1, Part 3)
8. Web2.Стендап 4.vtt (Standup 4)
9. Web2.П05.Спринт 2, частина 1.vtt (Sprint 2, Part 1)
10. Web2.Стендап 5.vtt (Standup 5)
11. Web2.П06.Спринт 2, частина 2.vtt (Sprint 2, Part 2)
12. Web2.Стендап 6.vtt (Standup 6)
13. Web2.П07.Спринт 2, частина 3.vtt (Sprint 2, Part 3)
14. Web2.Стендап 7.vtt (Standup 7)
15. Web2.П08.Спринт 3, частина 1.vtt (Sprint 3, Part 1)
16. Web2.Стендап 8.vtt (Standup 8)
17. Web2.П09.Спринт 3, частина 2.vtt (Sprint 3, Part 2)
18. Web2.Стендап 9.vtt (Standup 9)
19. Web2.П10.Спринт 3, частина 3.vtt (Sprint 3, Part 3)

**Pattern**: Introduction → (Sprint Part 1 → Standup) × 3 → (Sprint Part 2 → Standup) × 3 → (Sprint Part 3 → Standup) × 3
- Total: 1 Introduction + 9 Sprint sessions + 9 Stand-up sessions = 19 files
- File naming: `Web2.П[XX].[Description].vtt` or `Web2.Стендап [N].vtt`

## Technical Architecture

### Module Structure (Hybrid Architecture - Updated 2025-11-15)

**New Package Structure** (Option 3: Hybrid Approach):
```
draftcodebt1stprompt/
├── agile_education_analyzer/          # Main package directory (NEW)
│   ├── __init__.py                    # Package initialization, exports key classes
│   ├── data_structures.py             # Core data models (TranscriptSegment, etc.)
│   ├── ukrainian_patterns.py          # Ukrainian discourse pattern detection
│   ├── statistical_analysis.py        # Statistical tests and effect sizes
│   ├── visualization.py               # Publication-ready charts
│   ├── research_outputs.py            # LaTeX tables, quotations, APA formatting
│   └── utils/
│       ├── __init__.py
│       └── logger.py                  # Comprehensive logging system
│
├── tests/                             # Test suite (NEW)
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   └── test_ukrainian_patterns.py     # Ukrainian pattern tests
│
├── examples/                          # Jupyter notebooks (NEW)
│   └── quick_start.ipynb              # Quick start guide
│
├── setup.py                           # Package installation (NEW)
│
├── agile_education_analysis_framework.py  (1,343 lines) - LEGACY MONOLITHIC
│   (Maintained for backward compatibility during transition)
│
├── qualitative_coding.py  (750 lines) - QUALITATIVE ANALYSIS
│   ├── EducationalCodingScheme: Code definitions (Bloom's, interaction, etc.)
│   ├── QualitativeCoder: Apply codes to segments
│   ├── InterRaterReliability: Calculate agreement metrics
│   ├── CodeCoOccurrenceAnalyzer: Find code patterns
│   └── ThematicAnalyzer: Extract emergent themes
│
├── research_config.py  (221 lines) - CONFIGURATION
│   └── ResearchConfig: Centralized parameter management
│
└── run_analysis_example.py  (505 lines) - EXAMPLES
    └── Demonstration scripts for all features
```

**Key Improvements in New Architecture**:
- ✅ Modular design while preserving research cohesion
- ✅ Enhanced Ukrainian discourse detection (questions, confusion, confirmations)
- ✅ Comprehensive logging throughout (ResearchLogger for reproducibility)
- ✅ Research-ready outputs (LaTeX tables, APA formatting)
- ✅ Test suite with pytest
- ✅ Example Jupyter notebooks
- ✅ Proper Python package structure with setup.py
- ✅ Backward compatibility maintained

### Key Dependencies

**Critical Libraries**:
- **Ukrainian NLP**: `pymorphy3` (morphology), `stanza` (syntax), `spacy` (NER)
- **Embeddings**: `sentence-transformers` (multilingual models for clustering)
- **Topic Modeling**: `gensim` (LDA), `scikit-learn` (NMF)
- **Statistics**: `scipy` (tests), `statsmodels` (advanced stats)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `wordcloud`
- **Data**: `pandas`, `numpy`

**Important**: Some Ukrainian-specific models may need manual downloading or may not exist. Always have fallback strategies (e.g., multilingual models when Ukrainian unavailable).

## AI Assistant Guidelines

### When Working with This Repository

#### DO:

1. **Respect Research Integrity**:
   - Maintain proper statistical rigor (don't p-hack, report all tests)
   - Preserve qualitative coding schemes (don't arbitrarily change codes)
   - Keep methodological documentation updated
   - Ensure reproducibility (fixed random seeds, versioned data)

2. **Handle Ukrainian Text Carefully**:
   - Always specify `encoding='utf-8'` when reading/writing files
   - Test Unicode normalization (NFC vs NFD) if text looks corrupted
   - Be aware that string length != display width for Cyrillic
   - Don't assume English regex patterns work (e.g., `\w` behavior differs)

3. **Understand Educational Context**:
   - Student quotes should be anonymized (don't use real names in examples)
   - Challenges/problems should be framed constructively (not as failures)
   - Consider pedagogical implications of findings
   - Recognize power dynamics (teacher authority vs student voice)

4. **Follow Research Best Practices**:
   - Document all analysis decisions (parameters, thresholds, exclusions)
   - Report negative findings (not just significant results)
   - Include confidence intervals and effect sizes
   - Discuss limitations and threats to validity

5. **Code Quality**:
   - Maintain type hints (this is Python 3.8+ code)
   - Keep docstrings comprehensive and accurate
   - Add logging for debugging (don't use print statements)
   - Write modular, testable code

#### DON'T:

1. **Don't Assume English-Centric Approaches**:
   - Don't use English-only sentiment models
   - Don't apply English stop word lists
   - Don't ignore Cyrillic-specific encoding issues
   - Don't assume word tokenization works the same way

2. **Don't Violate Research Ethics**:
   - Don't expose student identities
   - Don't make value judgments about student performance
   - Don't alter data to fit expected outcomes
   - Don't skip ethical considerations in documentation

3. **Don't Break Methodological Rigor**:
   - Don't cherry-pick significant results
   - Don't run multiple tests without correction
   - Don't ignore assumptions of statistical tests
   - Don't mix qualitative and quantitative inappropriately

4. **Don't Oversimplify**:
   - Don't reduce complex educational phenomena to single metrics
   - Don't ignore contextual factors (class culture, external events)
   - Don't assume correlation implies causation
   - Don't generalize beyond this specific course/context

5. **Don't Create Unnecessary Files**:
   - Don't generate README files unless explicitly requested
   - Don't create documentation that duplicates existing guides
   - Don't add example notebooks without clear purpose
   - Don't clutter the repository with temporary outputs

### Common Tasks & How to Approach Them

#### Task 1: Adding a New Analysis Metric

**Example**: User wants to measure "collaborative problem-solving instances"

**Approach**:
1. **Understand the construct**: What does "collaborative problem-solving" mean in educational research? Look for multiple students working together on a problem.
2. **Define operationally**: Create measurable indicators (e.g., overlapping speech, pronoun usage "ми" [we], shared code references)
3. **Choose appropriate method**: Could be pattern matching, could be ML classification
4. **Validate**: Test on sample transcripts, calculate inter-rater reliability if human coding involved
5. **Integrate**: Add to appropriate analyzer class, update config, add visualization
6. **Document**: Explain rationale, limitations, interpretation guidelines

#### Task 2: Debugging Ukrainian NLP Issues

**Example**: Topic modeling produces gibberish topics

**Troubleshooting Steps**:
1. **Check encoding**: Verify UTF-8 throughout pipeline
2. **Inspect tokenization**: Ukrainian tokenization different from English
3. **Review stop words**: May need more comprehensive Ukrainian stop word list
4. **Examine preprocessing**: Lemmatization may be failing (check pymorphy3 installation)
5. **Validate corpus**: Too small/homogeneous corpus creates poor topics
6. **Adjust parameters**: Try different n_topics, min_df, max_df values
7. **Consider alternatives**: Try NMF instead of LDA, or increase n-gram range

#### Task 3: Extending Speaker Diarization

**Example**: Accuracy is low, need improvement

**Enhancement Strategy**:
1. **Analyze errors**: Where does it fail? Teacher/student confusion? Individual student identification?
2. **Add linguistic features**:
   - Teachers use imperatives, questions, formal address
   - Students use first person, uncertainty markers, informal language
3. **Improve clustering**:
   - Try different embedding models
   - Adjust clustering parameters (eps, min_samples for DBSCAN)
   - Add supervised learning if some labeled data available
4. **Manual validation**: Sample random segments, calculate accuracy, find patterns in errors
5. **Iterative refinement**: Adjust rules/models, re-test, repeat

#### Task 4: Running Complete Analysis on All Transcripts

**Workflow**:
1. **Setup environment**: Ensure all dependencies installed, models downloaded
2. **Validate data**: Check all 19 VTT files are accessible, properly encoded
3. **Configure analysis**: Review `research_config.py`, adjust parameters as needed
4. **Run in stages**:
   - First: VTT parsing and speaker diarization (save intermediate results)
   - Second: Engagement and agile adoption analysis
   - Third: Problem identification and sentiment analysis
   - Fourth: Statistical analysis and visualization
   - Fifth: Report generation
5. **Quality check**: Review outputs at each stage before proceeding
6. **Generate deliverables**: Research report, data exports, visualizations
7. **Archive**: Save all outputs with timestamps, document any anomalies

#### Task 5: Preparing for Publication

**Checklist**:
- [ ] Run all analyses with fixed random seeds (reproducibility)
- [ ] Generate high-resolution figures (300+ DPI for publication)
- [ ] Create supplementary materials (code, data dictionary, analysis scripts)
- [ ] Write methods section detailing all procedures
- [ ] Calculate and report all effect sizes
- [ ] Include limitations and future work sections
- [ ] Ensure anonymization throughout (no student names)
- [ ] Prepare data availability statement
- [ ] Consider pre-registration or open science framework
- [ ] Get ethics approval documentation ready

### Understanding the Research Questions

The framework is designed around four primary research questions (RQs). Any modifications should support answering these:

#### RQ1: Student Participation Evolution
**What's really being asked**: Do students become more engaged over time? Is there a "cold start" problem? Do some students dominate while others remain silent?

**Key Metrics**:
- `participation_rate`: What % of students contribute in each session?
- `utterance_count`: How often do students speak?
- `question_frequency`: Are students asking questions (active learning)?
- `proactive_contributions`: Unsolicited contributions vs responses to teacher questions

**Analytical Considerations**:
- Time series analysis (session as temporal unit)
- Within-student variation (some students may increase, others decrease)
- Sprint transitions (participation may dip at start of new sprint)
- Session type effects (stand-ups vs. sprint sessions)

#### RQ2: Agile Concept Understanding
**What's really being asked**: Are students actually learning agile, or just going through motions? What concepts are "sticky" vs. "slippery"?

**Key Metrics**:
- `terminology_adoption`: Do students use agile vocabulary spontaneously?
- `correct_usage_rate`: Context-appropriate application
- `misconception_frequency`: Common misunderstandings
- `concept_progression`: Which concepts learned early vs. late

**Analytical Considerations**:
- Distinguish between parroting and understanding (context matters)
- Track student-initiated vs. teacher-prompted usage
- Identify which concepts transfer from instruction to practice
- Map concept dependencies (understanding X requires understanding Y)

#### RQ3: Technical Challenges
**What's really being asked**: What are the actual pain points? Where do students get stuck? What support is needed?

**Key Metrics**:
- `problem_frequency`: How often do issues arise?
- `problem_categories`: Technical vs. conceptual vs. process
- `recurring_problems`: Same issue appearing multiple times
- `resolution_patterns`: How problems get solved (teacher help, peer support, independent)

**Analytical Considerations**:
- Some problems are "productive struggle" (desirable difficulty)
- Others are "blockers" (need intervention)
- Recurring problems may indicate curriculum design issues
- Problem co-occurrence (cascading failures)

#### RQ4: Stand-up Effectiveness
**What's really being asked**: Are stand-ups just ritual, or actually pedagogically valuable? Do they foster reflection and collaboration?

**Key Metrics**:
- `standup_engagement`: Participation levels in stand-ups
- `problem_identification`: Issues surfaced during stand-ups
- `peer_interaction`: Student-to-student dialogue
- `knowledge_sharing`: Students helping each other

**Analytical Considerations**:
- Compare stand-up metrics to regular session metrics
- Look for stand-up effects on subsequent sprint work
- Assess quality vs. quantity of contributions
- Identify stand-up "rituals" that may reduce effectiveness

### Code Reading Guide

When examining the codebase:

#### `agile_education_analysis_framework.py` (1,343 lines)

**Key Classes to Understand**:

1. **VTTProcessor** (lines ~50-200):
   - Handles WebVTT parsing with `webvtt-py`
   - Critical method: `parse_vtt_file()` - returns structured segments
   - Ukrainian text normalization happens here
   - Watermark removal patterns

2. **SpeakerDiarization** (lines ~200-400):
   - Pattern-based identification: regex for teacher markers
   - Embedding-based clustering: uses sentence-transformers
   - Critical method: `identify_speakers()` - assigns speaker roles
   - Accuracy depends heavily on transcript quality

3. **EngagementAnalyzer** (lines ~400-600):
   - Calculates participation metrics
   - Critical method: `calculate_engagement_scores()` - returns per-segment scores
   - Uses question detection (Ukrainian question markers: "?", "чи", "як", etc.)
   - Tracks confusion markers ("не розумію", "а що це")

4. **AgileAdoptionAnalyzer** (lines ~600-750):
   - Maintains agile terminology dictionary (Ukrainian and English)
   - Critical method: `analyze_terminology_usage()` - frequency and context
   - Identifies misconceptions through context analysis
   - Tracks adoption curves over sessions

5. **ProblemIdentifier** (lines ~750-900):
   - Pattern matching for problem indicators
   - Categories: technical, conceptual, process, collaboration, tools
   - Critical method: `identify_problems()` - extracts problem segments
   - Uses semantic similarity for finding recurring problems

6. **SentimentTopicAnalyzer** (lines ~900-1050):
   - Sentiment: tries Ukrainian models, falls back to multilingual
   - Topic modeling: LDA and NMF implementations
   - Critical method: `extract_topics()` - returns topics with top terms
   - Ukrainian stop words critical for quality

7. **StatisticalAnalyzer** (lines ~1050-1150):
   - Non-parametric tests (Kruskal-Wallis, Mann-Whitney)
   - Effect size calculations (Cohen's d)
   - Correlation analysis (Spearman for non-normal data)
   - Critical: applies Bonferroni correction for multiple comparisons

8. **ResearchVisualizer** (lines ~1150-1300):
   - Publication-quality figures (matplotlib + seaborn)
   - Interactive visualizations (plotly)
   - Critical: Ukrainian font support for labels
   - Exports PNG (high DPI) and SVG (vector)

9. **AgileEducationAnalyzer** (lines ~1300-1343):
   - Orchestration class - ties everything together
   - Critical method: `analyze_all_sessions()` - complete pipeline
   - Handles caching and intermediate results
   - Generates comprehensive markdown reports

#### `qualitative_coding.py` (750 lines)

**Key Classes to Understand**:

1. **EducationalCodingScheme** (lines ~50-250):
   - Static methods return code dictionaries
   - `bloom_taxonomy_codes()`: 6 cognitive levels
   - `interaction_pattern_codes()`: teacher-student dynamics
   - `agile_learning_codes()`: methodology-specific indicators
   - `problem_solving_codes()`: solution development process
   - Each code has: id, name, description, indicators (keywords/patterns)

2. **QualitativeCoder** (lines ~250-450):
   - Applies codes to transcript segments
   - Critical method: `auto_code()` - pattern matching for codes
   - Maintains code definitions and application rules
   - Supports both rule-based and ML-based coding

3. **InterRaterReliability** (lines ~450-550):
   - Calculates agreement metrics between coders
   - Cohen's Kappa: accounts for chance agreement
   - Krippendorff's Alpha: more robust for missing data
   - Critical for validating coding quality

4. **CodeCoOccurrenceAnalyzer** (lines ~550-650):
   - Identifies which codes appear together
   - Creates co-occurrence matrices
   - Finds code clusters (related concepts)
   - Useful for understanding code relationships

5. **ThematicAnalyzer** (lines ~650-750):
   - Extracts emergent themes from coded segments
   - Groups related codes into themes
   - Creates theme hierarchies
   - Critical for qualitative reporting

#### `research_config.py` (221 lines)

**Configuration Structure**:
```python
config = {
    'language': {
        'primary': 'uk',  # Ukrainian
        'fallback': 'multilingual',
        'encoding': 'utf-8'
    },
    'analysis': {
        'topic_modeling': {
            'n_topics': 10,
            'algorithm': 'lda',  # or 'nmf'
            'min_df': 2,
            'max_df': 0.8
        },
        'speaker_diarization': {
            'method': 'hybrid',  # pattern + embedding
            'embedding_model': 'distiluse-base-multilingual-cased-v2'
        },
        'engagement': {
            'question_weight': 2.0,
            'contribution_weight': 1.0
        }
    },
    'statistics': {
        'significance_level': 0.05,
        'bonferroni_correction': True,
        'effect_size_threshold': 0.5  # medium effect
    },
    'visualization': {
        'dpi': 300,
        'style': 'seaborn-v0_8-paper',
        'font_family': 'DejaVu Sans'  # Supports Cyrillic
    }
}
```

**Key Parameters to Adjust**:
- `language.fallback`: What to do when Ukrainian models unavailable
- `analysis.topic_modeling.n_topics`: More topics = more granular but potentially less coherent
- `statistics.significance_level`: 0.05 standard, 0.01 more conservative
- `visualization.font_family`: MUST support Cyrillic

#### `run_analysis_example.py` (505 lines)

**Example Workflows**:
1. **Basic analysis** (lines ~50-150): Load data, run analysis, generate report
2. **Custom coding** (lines ~150-250): Apply qualitative codes, calculate reliability
3. **Advanced statistics** (lines ~250-350): Hypothesis testing, effect sizes
4. **Visualization gallery** (lines ~350-450): All plot types demonstrated
5. **Export examples** (lines ~450-505): CSV, JSON, Excel formats

## Deep Insights & Philosophical Considerations

### The Nature of Educational Data

Transcripts are **not objective records** - they are:
- **Incomplete**: Not all learning happens verbally (students thinking, writing code)
- **Contextual**: Tone, body language, screen sharing lost in text
- **Socially constructed**: What gets said is shaped by classroom power dynamics
- **Temporally bounded**: Snapshot of one course, one semester, one instructor

**Implication for Analysis**: Quantitative metrics must be interpreted qualitatively. High participation ≠ high learning. Low question frequency could mean confusion OR confidence.

### The Ukrainian Language Dimension

Working with Ukrainian transcripts is not just a technical challenge - it's **epistemological**:
- **Linguistic relativity**: Some agile concepts may not translate cleanly (e.g., "accountability")
- **Code-switching**: Students switching to English for technical terms reveals conceptual boundaries
- **Cultural context**: Ukrainian educational culture (post-Soviet) has specific norms about student voice

**Implication for Analysis**: Don't just translate and apply English frameworks. Understand what Ukrainian patterns reveal about learning.

### Agile in Education Paradox

Teaching agile **using** agile creates interesting meta-levels:
- Students are learning agile by experiencing agile
- The classroom itself is a "team" undergoing sprints
- Assessment of learning IS the retrospective

**Implication for Analysis**: Traditional metrics (test scores) may miss the point. Look for process understanding, adaptability, collaboration.

### Ethical Considerations in Learning Analytics

This framework has **power** - it quantifies student behavior in ways that could:
- **Surveillance**: Make students feel monitored (chilling effect on participation)
- **Labeling**: Identify "low performers" in ways that become self-fulfilling
- **Decontextualization**: Reduce rich learning experiences to metrics
- **Misuse**: Used for punitive assessment rather than formative feedback

**Implication for Design**:
- Aggregate data to protect individuals
- Frame findings constructively (opportunities, not failures)
- Involve students in interpretation (member checking)
- Use for improvement, not judgment

## Future Directions & Extensibility

### Potential Enhancements

1. **Multimodal Analysis**:
   - Integrate video: gestures, facial expressions, screen sharing
   - Code repository analysis: link discussion to actual code produced
   - Combine transcript + git commits for holistic view

2. **Real-Time Applications**:
   - Live dashboard for instructors during class
   - Automated alerts for low engagement
   - Recommendation system for teaching strategies

3. **Comparative Research**:
   - Cross-institutional studies
   - Longitudinal tracking (same students across courses)
   - Agile vs. traditional pedagogy comparisons

4. **Advanced NLP**:
   - Fine-tune Ukrainian BERT for educational domain
   - Deep learning for speaker identification
   - Automated question classification (Bloom's levels)

### How to Extend

**Adding a New Analyzer Class**:
```python
class NewAnalyzer:
    """
    Description of what this analyzer does.

    Research context: Why is this analysis needed?
    Methodological basis: What theory/framework supports this?
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Initialize with configuration

    def analyze(self, segments: List[Segment]) -> Dict:
        """
        Main analysis method.

        Args:
            segments: List of transcript segments

        Returns:
            Dict with analysis results, including:
            - metrics: numerical results
            - patterns: identified patterns
            - metadata: analysis parameters used
        """
        # Implementation
        pass

    def visualize(self, results: Dict, output_path: str):
        """Generate visualization for this analysis."""
        pass
```

**Integration Steps**:
1. Add class to `agile_education_analysis_framework.py`
2. Add configuration parameters to `research_config.py`
3. Add example usage to `run_analysis_example.py`
4. Update `AgileEducationAnalyzer.analyze_all_sessions()` to call new analyzer
5. Add visualization to `ResearchVisualizer`
6. Document in `USAGE_GUIDE.md`

### Testing Strategy

**Unit Tests** (currently missing - opportunity for contribution):
```python
# tests/test_vtt_processor.py
def test_ukrainian_text_normalization():
    """Test that Ukrainian characters preserved correctly."""
    processor = VTTProcessor()
    text = "Привіт, як справи?"
    normalized = processor.normalize_text(text)
    assert "і" in normalized  # Ukrainian і, not i
    assert normalized.startswith("Привіт")

def test_speaker_diarization_accuracy():
    """Test speaker identification on known sample."""
    diarizer = SpeakerDiarization()
    sample_segments = load_test_data("sample.vtt")
    result = diarizer.identify_speakers(sample_segments)
    # Check against manually labeled ground truth
    accuracy = calculate_accuracy(result, ground_truth)
    assert accuracy > 0.80  # 80% minimum acceptable
```

**Integration Tests**:
- End-to-end pipeline on sample transcript
- Verify all outputs generated correctly
- Check encoding preservation throughout pipeline

**Validation Tests**:
- Compare auto-coding to manual coding (inter-rater reliability)
- Validate statistical assumptions (e.g., check for normality before t-test)
- Cross-validate topic models (coherence scores)

## Troubleshooting Guide

### Issue: "UnicodeDecodeError when reading VTT files"

**Cause**: File not UTF-8 encoded, or BOM (byte order mark) present

**Solutions**:
```python
# Try different encodings
for encoding in ['utf-8', 'utf-8-sig', 'cp1251']:  # cp1251 common for Ukrainian
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        break
    except UnicodeDecodeError:
        continue

# Remove BOM if present
content = content.lstrip('\ufeff')
```

### Issue: "Speaker diarization assigns all segments to one speaker"

**Cause**: Clustering algorithm not finding distinct groups

**Solutions**:
1. Check if transcript actually has speaker labels embedded
2. Adjust clustering parameters (lower min_samples)
3. Add more linguistic patterns for role identification
4. Try different embedding model
5. Manually validate sample - maybe it's actually a monologue

### Issue: "Topic modeling produces topics like: топік 1: та в і на з"

**Cause**: Stop words not removed, or too few content words

**Solutions**:
1. Expand Ukrainian stop word list (include "та", "в", "і", "на", "з")
2. Increase `min_df` (ignore very rare words)
3. Decrease `max_df` (ignore very common words)
4. Use n-grams (2-grams, 3-grams) to capture phrases
5. Check if lemmatization is working (should reduce inflected forms)

### Issue: "Statistical tests show no significant differences when expected"

**Cause**: Low power (small sample size), high variance, or inappropriate test

**Solutions**:
1. Check sample sizes (< 30 per group is often too small)
2. Calculate effect sizes (may have meaningful effects even if not significant)
3. Visualize distributions (box plots, histograms)
4. Consider within-subjects design if comparing same students over time
5. Check for violations of test assumptions
6. Try more powerful non-parametric tests

### Issue: "Sentiment analysis returns neutral for everything"

**Cause**: Ukrainian sentiment model not working, or educational language is genuinely neutral

**Solutions**:
1. Verify sentiment model actually loaded
2. Test on known Ukrainian sentences with clear sentiment
3. Check if model expects specific input format
4. Educational discourse IS often neutral - this may be accurate
5. Focus on specific emotions (confusion, frustration) rather than general sentiment
6. Consider fine-tuning model on educational corpus

## Conclusion: Working Principles

When working with this repository as an AI assistant:

1. **Respect the Research**: This is not just code - it's a research instrument. Methodological integrity matters.

2. **Embrace Complexity**: Educational phenomena are multifaceted. Resist oversimplification.

3. **Center Language**: Ukrainian is not an afterthought. It's central to the analysis.

4. **Prioritize Ethics**: Students' privacy and dignity come before analytical convenience.

5. **Think Critically**: Question assumptions, validate findings, consider alternatives.

6. **Document Thoroughly**: Future researchers (and future you) will thank you.

7. **Collaborate**: This is educational research - involve educators, students, domain experts.

8. **Iterate**: First analysis is never final. Refine, validate, improve.

9. **Communicate Clearly**: Research is only useful if communicated effectively.

10. **Remain Humble**: There's always more to learn about teaching, learning, and agile methodologies.

---

**This document is a living guide**. As the repository evolves, so should this document. When you add features, update this file. When you discover issues, document solutions here. When you have insights, capture them.

IMPORTANT THING FROM THE USER WHO CONTROL THIS REPOSITORY: document all steps that can help to clearly reproduce software creating, testing and the use to solve the global problem of analysing the research results for the future paper sections - Methodology, Results, and Limitations (all in LaTeX and JSON)


**Version**: 1.0
**Last Updated**: 2025-11-15
**Maintained By**: AI assistants working with this repository
**Philosophy**: Deep understanding enables better assistance
