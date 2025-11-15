"""
Qualitative Coding Module for Educational Research
Implements systematic coding schemes for analyzing educational transcripts
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Code:
    """Represents a single qualitative code"""
    id: str
    name: str
    description: str
    category: str
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    parent_code: Optional[str] = None
    
@dataclass
class CodedSegment:
    """Segment with applied codes"""
    text: str
    codes: List[str]
    timestamp: Optional[timedelta] = None
    speaker: Optional[str] = None
    confidence: float = 1.0
    coder_id: Optional[str] = None
    notes: Optional[str] = None

# ============================================================================
# CODING SCHEMES
# ============================================================================

class EducationalCodingScheme:
    """Predefined coding schemes for educational research"""
    
    @staticmethod
    def bloom_taxonomy_codes() -> List[Code]:
        """Bloom's Taxonomy cognitive levels"""
        return [
            Code(
                id='BLOOM_REMEMBER',
                name='Remembering',
                description='Recall facts and basic concepts',
                category='cognitive',
                keywords=['define', 'list', 'recall', 'name', 'identify'],
                patterns=[r'що таке', r'назвіть', r'згадайте', r'визначте']
            ),
            Code(
                id='BLOOM_UNDERSTAND',
                name='Understanding',
                description='Explain ideas or concepts',
                category='cognitive',
                keywords=['explain', 'describe', 'discuss', 'interpret'],
                patterns=[r'поясніть', r'опишіть', r'як ви розумієте', r'що означає']
            ),
            Code(
                id='BLOOM_APPLY',
                name='Applying',
                description='Use information in new situations',
                category='cognitive',
                keywords=['apply', 'use', 'implement', 'solve', 'demonstrate'],
                patterns=[r'застосуйте', r'використайте', r'вирішіть', r'покажіть']
            ),
            Code(
                id='BLOOM_ANALYZE',
                name='Analyzing',
                description='Draw connections among ideas',
                category='cognitive',
                keywords=['analyze', 'compare', 'contrast', 'examine'],
                patterns=[r'проаналізуйте', r'порівняйте', r'знайдіть різницю', r'дослідіть']
            ),
            Code(
                id='BLOOM_EVALUATE',
                name='Evaluating',
                description='Justify a stand or decision',
                category='cognitive',
                keywords=['evaluate', 'judge', 'critique', 'justify'],
                patterns=[r'оцініть', r'критикуйте', r'обґрунтуйте', r'доведіть']
            ),
            Code(
                id='BLOOM_CREATE',
                name='Creating',
                description='Produce new or original work',
                category='cognitive',
                keywords=['create', 'design', 'develop', 'construct'],
                patterns=[r'створіть', r'розробіть', r'спроектуйте', r'побудуйте']
            )
        ]
    
    @staticmethod
    def interaction_codes() -> List[Code]:
        """Teacher-student interaction patterns"""
        return [
            Code(
                id='INT_INITIATION',
                name='Initiation',
                description='Teacher initiates interaction',
                category='interaction',
                patterns=[r'давайте', r'хто може', r'скажіть']
            ),
            Code(
                id='INT_RESPONSE',
                name='Response',
                description='Student responds to teacher',
                category='interaction',
                patterns=[r'я думаю', r'мені здається', r'відповідь']
            ),
            Code(
                id='INT_FEEDBACK',
                name='Feedback',
                description='Teacher provides feedback',
                category='interaction',
                patterns=[r'правильно', r'молодець', r'не зовсім', r'спробуйте']
            ),
            Code(
                id='INT_QUESTION',
                name='Student Question',
                description='Student asks question',
                category='interaction',
                patterns=[r'\?', r'чому', r'як', r'можна запитати']
            ),
            Code(
                id='INT_PEER',
                name='Peer Interaction',
                description='Student-to-student interaction',
                category='interaction',
                patterns=[r'згоден з', r'додам до', r'як сказав']
            )
        ]
    
    @staticmethod
    def agile_learning_codes() -> List[Code]:
        """Agile methodology learning indicators"""
        return [
            Code(
                id='AGILE_CONCEPT',
                name='Agile Concept Understanding',
                description='Shows understanding of agile concepts',
                category='agile_learning',
                keywords=['sprint', 'standup', 'retrospective', 'backlog'],
                patterns=[r'спринт', r'стендап', r'ретроспектива', r'беклог']
            ),
            Code(
                id='AGILE_PRACTICE',
                name='Agile Practice Application',
                description='Applies agile practices',
                category='agile_learning',
                patterns=[r'планування спринту', r'daily standup', r'user story']
            ),
            Code(
                id='AGILE_MISCONCEPTION',
                name='Agile Misconception',
                description='Shows misconception about agile',
                category='agile_learning',
                patterns=[r'спринт.*місяць', r'ретро.*критика', r'скрам.*хаос']
            ),
            Code(
                id='AGILE_REFLECTION',
                name='Agile Process Reflection',
                description='Reflects on agile process',
                category='agile_learning',
                patterns=[r'процес допоміг', r'краще працювати', r'команда стала']
            )
        ]
    
    @staticmethod
    def problem_solving_codes() -> List[Code]:
        """Problem-solving behavior codes"""
        return [
            Code(
                id='PS_IDENTIFY',
                name='Problem Identification',
                description='Identifies a problem',
                category='problem_solving',
                patterns=[r'проблема', r'не працює', r'помилка', r'складність']
            ),
            Code(
                id='PS_ANALYZE',
                name='Problem Analysis',
                description='Analyzes problem causes',
                category='problem_solving',
                patterns=[r'тому що', r'причина', r'через те', r'викликано']
            ),
            Code(
                id='PS_SOLUTION',
                name='Solution Proposal',
                description='Proposes solution',
                category='problem_solving',
                patterns=[r'можна спробувати', r'пропоную', r'варіант', r'вирішити']
            ),
            Code(
                id='PS_IMPLEMENT',
                name='Solution Implementation',
                description='Implements solution',
                category='problem_solving',
                patterns=[r'зробив', r'виправив', r'змінив', r'додав']
            ),
            Code(
                id='PS_EVALUATE',
                name='Solution Evaluation',
                description='Evaluates solution effectiveness',
                category='problem_solving',
                patterns=[r'працює', r'допомогло', r'вирішено', r'результат']
            )
        ]

# ============================================================================
# QUALITATIVE CODER
# ============================================================================

class QualitativeCoder:
    """Main qualitative coding engine"""
    
    def __init__(self, coding_scheme: List[Code] = None):
        self.codes = {}
        self.coding_scheme = coding_scheme or []
        
        # Build code index
        for code in self.coding_scheme:
            self.codes[code.id] = code
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for code in self.coding_scheme:
            if code.patterns:
                self.compiled_patterns[code.id] = [
                    re.compile(pattern, re.IGNORECASE) 
                    for pattern in code.patterns
                ]
    
    def auto_code(self, text: str, speaker: str = None) -> List[str]:
        """Automatically apply codes based on patterns"""
        applied_codes = []
        text_lower = text.lower()
        
        for code_id, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    applied_codes.append(code_id)
                    break  # One match per code is enough
        
        # Apply additional heuristics based on speaker
        if speaker == 'Teacher':
            # Teachers more likely to initiate and give feedback
            if any(p.search(text_lower) for p in self.compiled_patterns.get('INT_INITIATION', [])):
                if 'INT_INITIATION' not in applied_codes:
                    applied_codes.append('INT_INITIATION')
        
        return applied_codes
    
    def manual_code(self, text: str, codes: List[str], 
                   coder_id: str = None, notes: str = None) -> CodedSegment:
        """Apply manual codes to a segment"""
        return CodedSegment(
            text=text,
            codes=codes,
            coder_id=coder_id,
            notes=notes,
            confidence=1.0  # Manual coding assumed high confidence
        )
    
    def suggest_codes(self, text: str, n_suggestions: int = 3) -> List[Tuple[str, float]]:
        """Suggest most likely codes for a segment"""
        scores = {}
        text_lower = text.lower()
        
        for code in self.coding_scheme:
            score = 0.0
            
            # Check patterns
            if code.id in self.compiled_patterns:
                for pattern in self.compiled_patterns[code.id]:
                    if pattern.search(text_lower):
                        score += 1.0
            
            # Check keywords
            for keyword in code.keywords:
                if keyword.lower() in text_lower:
                    score += 0.5
            
            if score > 0:
                scores[code.id] = score
        
        # Sort by score and return top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:n_suggestions]
    
    def validate_coding(self, coded_segments: List[CodedSegment]) -> Dict:
        """Validate coding consistency and coverage"""
        validation_results = {
            'total_segments': len(coded_segments),
            'coded_segments': 0,
            'uncoded_segments': 0,
            'codes_used': set(),
            'codes_unused': set(),
            'avg_codes_per_segment': 0,
            'code_frequency': Counter()
        }
        
        coded_count = 0
        total_codes = 0
        
        for segment in coded_segments:
            if segment.codes:
                coded_count += 1
                total_codes += len(segment.codes)
                validation_results['codes_used'].update(segment.codes)
                validation_results['code_frequency'].update(segment.codes)
            else:
                validation_results['uncoded_segments'] += 1
        
        validation_results['coded_segments'] = coded_count
        validation_results['codes_unused'] = set(self.codes.keys()) - validation_results['codes_used']
        
        if coded_count > 0:
            validation_results['avg_codes_per_segment'] = total_codes / coded_count
        
        return validation_results

# ============================================================================
# INTER-RATER RELIABILITY
# ============================================================================

class InterRaterReliability:
    """Calculate inter-rater reliability for qualitative coding"""
    
    @staticmethod
    def calculate_agreement(coder1_segments: List[CodedSegment], 
                          coder2_segments: List[CodedSegment],
                          code_list: List[str]) -> Dict:
        """Calculate various agreement metrics between two coders"""
        
        if len(coder1_segments) != len(coder2_segments):
            raise ValueError("Segment lists must have same length")
        
        results = {
            'percent_agreement': {},
            'cohen_kappa': {},
            'confusion_matrix': {}
        }
        
        # For each code, calculate agreement
        for code in code_list:
            coder1_binary = []
            coder2_binary = []
            
            for seg1, seg2 in zip(coder1_segments, coder2_segments):
                coder1_binary.append(1 if code in seg1.codes else 0)
                coder2_binary.append(1 if code in seg2.codes else 0)
            
            # Percent agreement
            agreements = sum(1 for c1, c2 in zip(coder1_binary, coder2_binary) if c1 == c2)
            results['percent_agreement'][code] = agreements / len(coder1_binary)
            
            # Cohen's Kappa
            if len(set(coder1_binary)) > 1 or len(set(coder2_binary)) > 1:
                kappa = cohen_kappa_score(coder1_binary, coder2_binary)
                results['cohen_kappa'][code] = kappa
            else:
                results['cohen_kappa'][code] = 1.0  # Perfect agreement if no variation
            
            # Confusion matrix
            tp = sum(1 for c1, c2 in zip(coder1_binary, coder2_binary) if c1 == 1 and c2 == 1)
            fp = sum(1 for c1, c2 in zip(coder1_binary, coder2_binary) if c1 == 1 and c2 == 0)
            fn = sum(1 for c1, c2 in zip(coder1_binary, coder2_binary) if c1 == 0 and c2 == 1)
            tn = sum(1 for c1, c2 in zip(coder1_binary, coder2_binary) if c1 == 0 and c2 == 0)
            
            results['confusion_matrix'][code] = {
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn
            }
        
        # Overall statistics
        results['overall'] = {
            'avg_percent_agreement': np.mean(list(results['percent_agreement'].values())),
            'avg_cohen_kappa': np.mean(list(results['cohen_kappa'].values())),
            'codes_with_perfect_agreement': sum(1 for k in results['cohen_kappa'].values() if k == 1.0),
            'codes_with_substantial_agreement': sum(1 for k in results['cohen_kappa'].values() if k > 0.6)
        }
        
        return results

# ============================================================================
# CODE CO-OCCURRENCE ANALYSIS
# ============================================================================

class CodeCoOccurrenceAnalyzer:
    """Analyze co-occurrence patterns in coded data"""
    
    def __init__(self, coded_segments: List[CodedSegment]):
        self.segments = coded_segments
        self.co_occurrence_matrix = None
        
    def calculate_co_occurrence(self) -> pd.DataFrame:
        """Calculate co-occurrence matrix for codes"""
        # Get all unique codes
        all_codes = set()
        for segment in self.segments:
            all_codes.update(segment.codes)
        
        all_codes = sorted(list(all_codes))
        
        # Initialize co-occurrence matrix
        matrix = pd.DataFrame(0, index=all_codes, columns=all_codes)
        
        # Count co-occurrences
        for segment in self.segments:
            codes = segment.codes
            for i, code1 in enumerate(codes):
                for code2 in codes[i:]:
                    matrix.loc[code1, code2] += 1
                    if code1 != code2:
                        matrix.loc[code2, code1] += 1
        
        self.co_occurrence_matrix = matrix
        return matrix
    
    def find_code_clusters(self, threshold: float = 0.5) -> List[Set[str]]:
        """Find clusters of frequently co-occurring codes"""
        if self.co_occurrence_matrix is None:
            self.calculate_co_occurrence()
        
        # Normalize matrix
        matrix_norm = self.co_occurrence_matrix.div(
            self.co_occurrence_matrix.max(axis=1), axis=0
        )
        
        # Find clusters using simple threshold
        clusters = []
        processed = set()
        
        for code1 in matrix_norm.index:
            if code1 in processed:
                continue
            
            cluster = {code1}
            processed.add(code1)
            
            for code2 in matrix_norm.columns:
                if code2 not in processed and matrix_norm.loc[code1, code2] > threshold:
                    cluster.add(code2)
                    processed.add(code2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def analyze_temporal_patterns(self) -> pd.DataFrame:
        """Analyze how code usage changes over time"""
        temporal_data = []
        
        for segment in self.segments:
            if segment.timestamp:
                for code in segment.codes:
                    temporal_data.append({
                        'timestamp': segment.timestamp.total_seconds(),
                        'code': code,
                        'speaker': segment.speaker
                    })
        
        df = pd.DataFrame(temporal_data)
        
        # Create time windows (e.g., 5-minute intervals)
        if not df.empty:
            df['time_window'] = pd.cut(df['timestamp'], bins=20, labels=False)
            
            # Count codes per time window
            temporal_patterns = df.groupby(['time_window', 'code']).size().reset_index(name='count')
            return temporal_patterns
        
        return pd.DataFrame()

# ============================================================================
# THEMATIC ANALYSIS
# ============================================================================

class ThematicAnalyzer:
    """Perform thematic analysis on coded data"""
    
    def __init__(self, coded_segments: List[CodedSegment], codes: Dict[str, Code]):
        self.segments = coded_segments
        self.codes = codes
        
    def identify_themes(self) -> List[Dict]:
        """Identify emergent themes from coded data"""
        # Group segments by code categories
        category_segments = defaultdict(list)
        
        for segment in self.segments:
            for code_id in segment.codes:
                if code_id in self.codes:
                    category = self.codes[code_id].category
                    category_segments[category].append(segment)
        
        # Analyze each category for themes
        themes = []
        
        for category, segments in category_segments.items():
            # Extract key phrases and concepts
            text_combined = ' '.join([s.text for s in segments])
            
            theme = {
                'category': category,
                'segment_count': len(segments),
                'codes_involved': list(set(c for s in segments for c in s.codes)),
                'key_phrases': self._extract_key_phrases(text_combined),
                'example_segments': segments[:3]  # First 3 examples
            }
            
            themes.append(theme)
        
        return themes
    
    def _extract_key_phrases(self, text: str, n_phrases: int = 5) -> List[str]:
        """Extract key phrases from text (simplified version)"""
        # This is a simplified implementation
        # In practice, use more sophisticated NLP techniques
        
        words = text.lower().split()
        # Create bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        # Count frequency
        phrase_counts = Counter(bigrams)
        
        # Filter out common phrases
        common_phrases = {'це є', 'я думаю', 'ми можемо', 'давайте розглянемо'}
        filtered_phrases = {p: c for p, c in phrase_counts.items() 
                          if p not in common_phrases and c > 1}
        
        # Return top N
        top_phrases = sorted(filtered_phrases.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in top_phrases[:n_phrases]]
    
    def create_theme_hierarchy(self) -> Dict:
        """Create hierarchical structure of themes"""
        hierarchy = {
            'root': {
                'name': 'All Themes',
                'children': []
            }
        }
        
        themes = self.identify_themes()
        
        # Group themes by category
        for theme in themes:
            category_node = {
                'name': theme['category'],
                'segment_count': theme['segment_count'],
                'children': []
            }
            
            # Add codes as children
            for code_id in theme['codes_involved']:
                if code_id in self.codes:
                    code = self.codes[code_id]
                    code_node = {
                        'name': code.name,
                        'id': code.id,
                        'description': code.description
                    }
                    category_node['children'].append(code_node)
            
            hierarchy['root']['children'].append(category_node)
        
        return hierarchy

# ============================================================================
# EXPORT AND REPORTING
# ============================================================================

class QualitativeReporter:
    """Generate reports from qualitative coding analysis"""
    
    def __init__(self, coded_segments: List[CodedSegment], codes: Dict[str, Code]):
        self.segments = coded_segments
        self.codes = codes
        
    def generate_summary_report(self) -> str:
        """Generate markdown summary report"""
        report = []
        report.append("# Qualitative Coding Analysis Report\n\n")
        
        # Basic statistics
        report.append("## Summary Statistics\n")
        report.append(f"- Total segments analyzed: {len(self.segments)}\n")
        
        coded_segments = [s for s in self.segments if s.codes]
        report.append(f"- Coded segments: {len(coded_segments)}\n")
        report.append(f"- Uncoded segments: {len(self.segments) - len(coded_segments)}\n")
        
        # Code frequency
        code_freq = Counter()
        for segment in self.segments:
            code_freq.update(segment.codes)
        
        report.append("\n## Most Frequent Codes\n")
        for code_id, count in code_freq.most_common(10):
            if code_id in self.codes:
                code_name = self.codes[code_id].name
                report.append(f"- {code_name} ({code_id}): {count} occurrences\n")
        
        # Category distribution
        category_counts = defaultdict(int)
        for segment in self.segments:
            for code_id in segment.codes:
                if code_id in self.codes:
                    category_counts[self.codes[code_id].category] += 1
        
        report.append("\n## Category Distribution\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {category}: {count} codes applied\n")
        
        # Speaker analysis
        speaker_codes = defaultdict(Counter)
        for segment in self.segments:
            if segment.speaker:
                speaker_codes[segment.speaker].update(segment.codes)
        
        report.append("\n## Speaker Analysis\n")
        for speaker, codes in speaker_codes.items():
            report.append(f"\n### {speaker}\n")
            report.append(f"- Total codes: {sum(codes.values())}\n")
            report.append(f"- Most common codes:\n")
            for code_id, count in codes.most_common(3):
                if code_id in self.codes:
                    report.append(f"  - {self.codes[code_id].name}: {count}\n")
        
        return ''.join(report)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export coded segments to pandas DataFrame"""
        data = []
        
        for segment in self.segments:
            row = {
                'text': segment.text,
                'codes': '|'.join(segment.codes),
                'num_codes': len(segment.codes),
                'speaker': segment.speaker,
                'timestamp': segment.timestamp.total_seconds() if segment.timestamp else None,
                'confidence': segment.confidence,
                'coder': segment.coder_id,
                'notes': segment.notes
            }
            
            # Add binary columns for each code
            for code_id in self.codes.keys():
                row[f'has_{code_id}'] = 1 if code_id in segment.codes else 0
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_for_statistical_analysis(self) -> Dict:
        """Export data formatted for statistical analysis"""
        export_data = {
            'segments': [],
            'codes': list(self.codes.keys()),
            'code_matrix': [],
            'metadata': {
                'total_segments': len(self.segments),
                'code_definitions': {}
            }
        }
        
        # Create binary matrix
        for segment in self.segments:
            row = [1 if code in segment.codes else 0 for code in export_data['codes']]
            export_data['code_matrix'].append(row)
            
            export_data['segments'].append({
                'text': segment.text[:100],  # First 100 chars
                'speaker': segment.speaker,
                'timestamp': segment.timestamp.total_seconds() if segment.timestamp else None
            })
        
        # Add code definitions
        for code_id, code in self.codes.items():
            export_data['metadata']['code_definitions'][code_id] = {
                'name': code.name,
                'category': code.category,
                'description': code.description
            }
        
        return export_data

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize coding schemes
    all_codes = (
        EducationalCodingScheme.bloom_taxonomy_codes() +
        EducationalCodingScheme.interaction_codes() +
        EducationalCodingScheme.agile_learning_codes() +
        EducationalCodingScheme.problem_solving_codes()
    )
    
    # Create coder instance
    coder = QualitativeCoder(all_codes)
    
    # Example text segments
    example_segments = [
        "Давайте розглянемо, що таке спринт у Scrum методології.",
        "Я не зовсім розумію, чому спринт має бути таким коротким?",
        "Спринт це ітерація розробки, яка зазвичай триває 2-4 тижні.",
        "На моїй попередній роботі ми мали спринти по місяцю, і це було зручно.",
        "Молодець! Ви правильно визначили основні характеристики спринту."
    ]
    
    # Auto-code segments
    coded_segments = []
    for text in example_segments:
        codes = coder.auto_code(text)
        segment = CodedSegment(text=text, codes=codes)
        coded_segments.append(segment)
        print(f"Text: {text[:50]}...")
        print(f"Codes: {codes}\n")
    
    # Validation
    validation = coder.validate_coding(coded_segments)
    print("Validation Results:")
    print(f"- Coded segments: {validation['coded_segments']}/{validation['total_segments']}")
    print(f"- Unique codes used: {len(validation['codes_used'])}")
    print(f"- Average codes per segment: {validation['avg_codes_per_segment']:.2f}")
    
    # Generate report
    reporter = QualitativeReporter(coded_segments, coder.codes)
    report = reporter.generate_summary_report()
    print("\n" + report[:500] + "...")
