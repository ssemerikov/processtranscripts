# ProcessTranscripts Repository - Structure and Goals

## Repository Overview

This repository contains a comprehensive research framework for analyzing educational transcripts from an agile web programming course taught in Ukrainian. The project focuses on extracting meaningful insights from video subtitle files (VTT format) to evaluate the effectiveness of agile methodologies in teaching web development.

**Primary Language**: Ukrainian (transcripts) / Python (analysis code)
**Research Domain**: Educational Research, Software Engineering Education, Agile Methodologies
**Current Status**: Analysis framework implemented, ready for transcript processing

---

## Repository Structure

```
processtranscripts/
├── .git/                           # Git version control
├── draftcodebt1stprompt/          # Python analysis framework
│   ├── agile_education_analysis_framework.py  (1,343 lines)
│   ├── qualitative_coding.py                  (750 lines)
│   ├── research_config.py                     (221 lines)
│   ├── run_analysis_example.py                (505 lines)
│   ├── requirements.txt                       # Python dependencies
│   ├── USAGE_GUIDE.md                         # Comprehensive usage documentation
│   └── readme.md                              # Brief description
├── transcripts/                    # Raw data: VTT transcript files
│   ├── Web2.П01.Вступ до гнучкої розробки.vtt
│   ├── Web2.П02-П10.Спринт [1-3], частина [1-3].vtt  (9 files)
│   ├── Web2.Стендап [1-9].vtt                         (9 files)
│   └── readme.md
├── prompt1.md                      # Initial research design prompt
├── prompt2.md                      # Technical implementation prompt
└── README.md                       # (to be created)
```

### Directory Details

#### `/draftcodebt1stprompt/` - Analysis Framework
This directory contains the complete Python-based analysis system for processing educational transcripts.

**Key Components:**

1. **`agile_education_analysis_framework.py`** (1,343 lines)
   - Main analysis framework
   - VTT file parsing and Ukrainian text processing
   - Speaker diarization (teacher/student identification)
   - Engagement metrics calculation
   - Agile terminology adoption tracking
   - Problem identification and pattern analysis
   - Sentiment and topic analysis
   - Statistical analysis tools
   - Visualization generation
   - Research report generation

2. **`qualitative_coding.py`** (750 lines)
   - Qualitative research coding schemes
   - Bloom's Taxonomy implementation
   - Interaction pattern codes
   - Agile learning codes
   - Problem-solving codes
   - Inter-rater reliability calculations
   - Code co-occurrence analysis
   - Thematic analysis tools

3. **`research_config.py`** (221 lines)
   - Configuration management
   - Analysis parameters
   - Ukrainian language settings
   - Statistical thresholds
   - Visualization preferences

4. **`run_analysis_example.py`** (505 lines)
   - Example usage scripts
   - Complete analysis pipeline
   - Individual component demonstrations
   - Export and reporting examples

5. **`requirements.txt`**
   - Python dependencies including:
     - Ukrainian NLP: spaCy, Stanza, pymorphy3
     - ML/Transformers: sentence-transformers, torch
     - Analysis: scikit-learn, gensim, nltk
     - Statistics: scipy, statsmodels
     - Visualization: matplotlib, seaborn, plotly, wordcloud

6. **`USAGE_GUIDE.md`** (455 lines)
   - Comprehensive usage documentation
   - Installation instructions
   - Feature descriptions
   - Code examples
   - Research methodology guidance
   - Troubleshooting tips

#### `/transcripts/` - Research Data
Contains 19 VTT (WebVTT) subtitle files from educational sessions:

**Session Types:**
- **1 Introduction Session**: "Вступ до гнuchкої розробки" (Introduction to Agile Development)
- **9 Sprint Sessions**: 3 sprints × 3 parts each
  - Sprint 1: Parts 1-3 (П02-П04)
  - Sprint 2: Parts 1-3 (П05-П07)
  - Sprint 3: Parts 1-3 (П08-П10)
- **9 Stand-up Sessions**: "Стендап 1-9"

**Transcript Characteristics:**
- Format: WebVTT (VTT) subtitle files
- Language: Ukrainian (with occasional English technical terms)
- Content: Live classroom recordings of agile web programming course
- Participants: Teacher(s) and students
- Timestamps: Preserved for temporal analysis

#### Root Level Files

1. **`prompt1.md`** (2,058 bytes)
   - Initial research design prompt
   - Outlines research questions and methodology
   - Defines key metrics and analysis approaches
   - Specifies visualization requirements

2. **`prompt2.md`** (2,375 bytes)
   - Technical implementation specifications
   - Module architecture requirements
   - Library selections and justifications
   - File structure design

---

## Research Goals and Objectives

### Primary Research Goals

1. **Evaluate Agile Methodology Effectiveness in Education**
   - Assess how well agile methods translate to educational contexts
   - Identify successful teaching patterns
   - Document challenges and barriers

2. **Measure Student Learning Outcomes**
   - Track comprehension of agile concepts
   - Monitor skill development across sprints
   - Evaluate knowledge retention and application

3. **Analyze Teaching Effectiveness**
   - Identify effective pedagogical strategies
   - Evaluate instructional clarity and engagement
   - Assess formative assessment techniques

4. **Document Student Engagement Patterns**
   - Measure participation levels
   - Track question-asking behavior
   - Analyze collaboration in stand-ups

### Specific Research Questions

#### RQ1: Student Participation Evolution
**Question**: How does student participation evolve across sprints?

**Metrics**:
- Participation rate (% of active contributors)
- Question frequency per student
- Utterance count and length
- Proactive vs. reactive contributions

**Expected Insights**:
- Participation trends over time
- Correlation between engagement and sprint progression
- Identification of highly engaged vs. passive students

#### RQ2: Agile Concept Understanding
**Question**: What agile concepts are most/least understood by students?

**Metrics**:
- Correct terminology usage rate
- Misconception frequency
- Context-appropriate application
- Vocabulary adoption timeline

**Expected Insights**:
- Which agile concepts are easiest/hardest to grasp
- Common misconceptions and misunderstandings
- Effectiveness of different teaching approaches

#### RQ3: Technical Challenges
**Question**: What technical challenges emerge repeatedly?

**Metrics**:
- Problem occurrence frequency
- Problem categories (technical, conceptual, process)
- Resolution patterns
- Recurring vs. novel issues

**Expected Insights**:
- Common stumbling blocks in agile education
- Technical prerequisites needed
- Support resources required

#### RQ4: Stand-up Effectiveness
**Question**: How effective are stand-ups for student learning?

**Metrics**:
- Stand-up engagement levels
- Problem identification rate
- Peer-to-peer interaction frequency
- Knowledge sharing instances

**Expected Insights**:
- Stand-up meeting effectiveness
- Student comfort with agile ceremonies
- Collaboration quality

---

## Technical Capabilities

### Data Processing Features

1. **VTT Parsing**
   - WebVTT format handling
   - UTF-8 Ukrainian text normalization
   - Timestamp preservation
   - Transcription error correction

2. **Speaker Diarization**
   - Pattern-based speaker identification
   - Teacher vs. student classification
   - Clustering using multilingual embeddings
   - Participation tracking per speaker

3. **Ukrainian NLP**
   - Morphological analysis (pymorphy3)
   - Named entity recognition
   - Lemmatization and tokenization
   - Part-of-speech tagging

### Analysis Methodologies

1. **Quantitative Analysis**
   - Participation statistics
   - Terminology frequency analysis
   - Temporal pattern detection
   - Correlation analysis

2. **Qualitative Analysis**
   - Bloom's Taxonomy coding
   - Interaction pattern coding
   - Problem-solving process coding
   - Thematic analysis

3. **Statistical Testing**
   - Kruskal-Wallis H-test (multi-group comparison)
   - Mann-Whitney U-test (pairwise comparison)
   - Cohen's d (effect size)
   - Bonferroni correction (multiple testing)

4. **Topic Modeling**
   - Latent Dirichlet Allocation (LDA)
   - Non-negative Matrix Factorization (NMF)
   - Topic evolution tracking
   - N-gram phrase extraction

5. **Sentiment Analysis**
   - Ukrainian sentiment models
   - Multilingual fallback options
   - Emotional tone tracking
   - Confusion/frustration detection

### Visualization Outputs

1. **Engagement Visualizations**
   - Line plots: participation over time
   - Bar charts: question frequency
   - Heatmaps: engagement by session

2. **Agile Adoption Visualizations**
   - Terminology usage trends
   - Adoption rate curves
   - Correct vs. incorrect usage

3. **Problem Analysis Visualizations**
   - Problem category heatmaps
   - Recurring issue tracking
   - Session × problem matrices

4. **Network Visualizations**
   - Speaker interaction networks
   - Question-answer flows
   - Collaboration patterns

5. **Word Clouds**
   - Sprint-specific vocabulary
   - Technical term frequency
   - Student vs. teacher language

### Export and Reporting

**Output Formats**:
- JSON: Structured analysis results
- CSV: Tabular data for further analysis
- Excel: Multi-sheet workbooks
- Markdown: Human-readable research reports
- PNG/SVG: Publication-ready visualizations

---

## Possible Future Goals

### Short-term Enhancements

1. **Run Complete Analysis**
   - Process all 19 VTT transcripts
   - Generate comprehensive research report
   - Produce publication-ready visualizations
   - Export data for statistical software (R, SPSS)

2. **Validation and Refinement**
   - Manual validation of speaker diarization
   - Inter-rater reliability testing for coding
   - Qualitative finding triangulation
   - Statistical assumption verification

3. **Documentation Expansion**
   - Create detailed README.md
   - Add code documentation
   - Include example outputs
   - Write research methodology appendix

### Medium-term Research Extensions

1. **Comparative Studies**
   - Compare with traditional (non-agile) courses
   - Cross-institution analysis
   - Different programming languages/domains
   - Various agile frameworks (Scrum, Kanban, XP)

2. **Longitudinal Analysis**
   - Track students across multiple courses
   - Alumni career outcome correlation
   - Long-term skill retention
   - Industry preparedness assessment

3. **Advanced Analytics**
   - Deep learning for speaker recognition
   - Automated question classification
   - Predictive modeling (at-risk students)
   - Real-time engagement feedback

4. **Multimodal Analysis**
   - Video analysis (gestures, screen sharing)
   - Code repository analysis (GitHub commits)
   - Project artifact evaluation
   - Combined transcript + code analysis

### Long-term Vision

1. **Educational Tool Development**
   - Real-time engagement dashboard for instructors
   - Automated feedback generation
   - Student self-assessment tools
   - Teaching recommendation system

2. **Scalability**
   - Multi-language support (beyond Ukrainian)
   - Large-scale course analysis
   - MOOC transcript processing
   - Cross-platform integration

3. **Publication and Dissemination**
   - Academic papers in CS education journals
   - Conference presentations
   - Open-source toolkit release
   - Educational practitioner guides

4. **Community Building**
   - Collaborate with agile education researchers
   - Share datasets (with privacy considerations)
   - Develop standardized metrics
   - Create benchmark datasets

---

## Technical Dependencies

### Core Requirements
- **Python**: 3.8+
- **Operating System**: Linux/macOS/Windows with UTF-8 support
- **Memory**: 8GB+ RAM recommended for large-scale analysis
- **Storage**: ~500MB for dependencies, variable for outputs

### Key Libraries

**NLP & Ukrainian Language**:
- spaCy, Stanza (syntactic analysis)
- pymorphy3 (morphological analysis)
- sentence-transformers (embeddings)
- transformers (BERT models)

**Machine Learning**:
- scikit-learn (clustering, classification)
- gensim (topic modeling)
- torch (deep learning backend)

**Statistical Analysis**:
- scipy (hypothesis testing)
- statsmodels (advanced statistics)
- pandas (data manipulation)
- numpy (numerical computing)

**Visualization**:
- matplotlib, seaborn (static plots)
- plotly (interactive visualizations)
- wordcloud (text visualization)
- networkx (graph visualization)

---

## Data Privacy and Ethics

### Considerations

1. **Participant Consent**
   - Ensure students consented to recording and analysis
   - Anonymize speaker identities in reports
   - Protect sensitive information

2. **Data Storage**
   - Secure storage of transcripts
   - Access control for research data
   - Compliance with institutional policies

3. **Reporting**
   - Aggregate findings to prevent individual identification
   - Ethical presentation of student difficulties
   - Constructive framing of challenges

---

## Getting Started

### For Researchers

1. **Installation**
   ```bash
   cd draftcodebt1stprompt
   pip install -r requirements.txt
   python -m spacy download uk_core_news_sm
   python -m stanza.download uk
   ```

2. **Basic Analysis**
   ```python
   from agile_education_analysis_framework import AgileEducationAnalyzer

   analyzer = AgileEducationAnalyzer('../transcripts')
   results = analyzer.analyze_all_sessions()
   analyzer.generate_report(results, 'report.md')
   ```

3. **Consult Documentation**
   - Read `USAGE_GUIDE.md` for detailed instructions
   - Review `run_analysis_example.py` for usage patterns
   - Check `research_config.py` for customization options

### For Developers

1. **Code Structure**
   - Main framework: `agile_education_analysis_framework.py`
   - Qualitative analysis: `qualitative_coding.py`
   - Configuration: `research_config.py`
   - Examples: `run_analysis_example.py`

2. **Extension Points**
   - Add custom coding schemes in `EducationalCodingScheme`
   - Extend analysis modules (create new analyzer classes)
   - Add visualization types in `ResearchVisualizer`
   - Implement custom metrics

3. **Testing**
   - Test with individual VTT files first
   - Validate Ukrainian text processing
   - Verify statistical calculations
   - Check visualization outputs

---

## Repository Metadata

**Created**: 2024 (based on commit history)
**Language**: Python 3.8+ (code), Ukrainian (data)
**Domain**: Educational Research, Computer Science Education
**Methodology**: Mixed methods (quantitative + qualitative)
**Total Code**: ~2,800 lines of Python
**Data Files**: 19 VTT transcripts
**Documentation**: ~15,000 words

---

## Contributing

This repository represents a specific research project. For contributions:

1. **Bug Reports**: Document issues with Ukrainian text processing or analysis errors
2. **Feature Requests**: Suggest additional metrics or analysis methods
3. **Code Improvements**: Optimize performance, enhance documentation
4. **Research Collaboration**: Contact repository owner for collaborative opportunities

---

## License and Citation

**License**: TBD (consider MIT or GPL for code, appropriate license for data)

**Suggested Citation**:
```
[Author]. (2024). Agile Education Analysis Framework for Ukrainian Transcripts.
GitHub repository: processtranscripts
```

---

## Conclusion

This repository provides a comprehensive framework for analyzing agile software engineering education through transcript analysis. It combines advanced NLP techniques, educational research methodologies, and statistical rigor to extract meaningful insights from Ukrainian-language classroom recordings.

The framework is designed to be:
- **Extensible**: Easy to add new analysis modules
- **Reproducible**: Clear configuration and documentation
- **Rigorous**: Based on established research methodologies
- **Practical**: Generates actionable insights for educators

The ultimate goal is to improve agile software engineering education by understanding how students learn agile methods, what challenges they face, and how instructors can teach more effectively.

---

**Last Updated**: 2025-11-15
**Repository**: processtranscripts
**Status**: Active Development
