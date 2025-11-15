I need to develop a Python application to analyze educational transcripts from an agile web programming course. The data consists of 19 VTT files in Ukrainian. Please create a modular Python system with the following components:

Technical Requirements:
- Parse VTT files preserving timestamps and speaker information
- Handle Ukrainian text (UTF-8, tokenization, lemmatization)
- Extract and analyze conversation patterns
- Identify individual speakers and track their participation

Create Python code with these modules:

1. VTT Parser Module:
   - Extract text, timestamps, and potential speaker markers
   - Clean and normalize Ukrainian text
   - Handle VTT format variations

2. Speaker Identification Module:
   - Detect speaker changes
   - Create speaker profiles
   - Track participation metrics per speaker

3. Content Analysis Module:
   - Keyword extraction (agile terms, technical terms, questions)
   - Pattern detection (repeated issues, common questions)
   - Sentiment/emotion analysis adapted for educational context
   - Topic segmentation by timestamp

4. Agile-Specific Analysis:
   - Sprint progression tracking
   - Stand-up pattern analysis (blockers, progress, plans)
   - Agile terminology frequency and context
   - Team dynamics indicators

5. Educational Metrics Module:
   - Question frequency and types
   - Response patterns
   - Engagement scoring
   - Knowledge progression indicators

6. Statistical Analysis Module:
   - Participation statistics per session
   - Temporal analysis (activity over time)
   - Correlation between participation and sprint progression
   - Comparative analysis across sprints

7. Export Module:
   - JSON/CSV export for quantitative data
   - Markdown reports with key findings
   - Visualization exports (matplotlib/plotly)

Libraries to use:
- webvtt-py for VTT parsing
- pymorphy2 or stanza for Ukrainian NLP
- pandas for data manipulation
- scikit-learn for ML analysis
- matplotlib/seaborn for visualization

Include error handling, logging, and configuration options. Make the code extensible for future transcript additions. Add docstrings and type hints.

File structure:
transcripts/
├── parser/
├── nlp/
├── analysis/
├── visualization/
├── reports/
└── main.py

Start with the core VTT parser and speaker identification modules, then build the analysis layers.
