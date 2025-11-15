"""
Comprehensive Analysis Framework for Agile Education Research
Educational Research on Agile Methodologies in Web Programming Instruction
Author: Research Framework Generator
Date: November 2024
Language: Ukrainian transcript analysis
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import unicodedata

# Data Processing Libraries
import pandas as pd
import numpy as np
import webvtt
from dateutil import parser

# NLP Libraries for Ukrainian
import spacy
import stanza
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import pymorphy3  # Ukrainian morphological analyzer
from ukrainian_word_stress import Stressifier  # Optional for prosody analysis

# Text Analysis Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from gensim import corpora, models
from gensim.models import Word2Vec, CoherenceModel
import nltk
from textblob import TextBlob

# Statistical Analysis
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, friedmanchisquare
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx

# Qualitative Analysis
from qualitative_coding import QualitativeCoder  # Custom implementation needed

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA STRUCTURES AND CONFIGURATION
# ============================================================================

@dataclass
class TranscriptSegment:
    """Represents a single segment from VTT file"""
    index: int
    start_time: timedelta
    end_time: timedelta
    text: str
    speaker: Optional[str] = None
    sentiment: Optional[float] = None
    engagement_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    
@dataclass
class SessionMetadata:
    """Metadata for each session"""
    filename: str
    session_type: str  # 'introduction', 'sprint', 'standup'
    sprint_number: Optional[int] = None
    part_number: Optional[int] = None
    duration: Optional[timedelta] = None
    participant_count: Optional[int] = None
    date: Optional[datetime] = None

@dataclass
class AgileTerminology:
    """Agile terminology patterns for Ukrainian"""
    terms = {
        'sprint': ['спринт', 'спрінт', 'sprint'],
        'standup': ['стендап', 'станд-ап', 'stand-up', 'щоденна зустріч'],
        'backlog': ['беклог', 'бек-лог', 'backlog', 'список завдань'],
        'scrum': ['скрам', 'scrum'],
        'agile': ['гнучка розробка', 'гнучкий', 'agile', 'аджайл'],
        'user_story': ['користувацька історія', 'user story', 'юзер сторі'],
        'retrospective': ['ретроспектива', 'ретро', 'retrospective'],
        'velocity': ['швидкість', 'velocity', 'продуктивність команди'],
        'burndown': ['burndown', 'графік згоряння', 'берндаун'],
        'iteration': ['ітерація', 'iteration'],
        'kanban': ['канбан', 'kanban'],
        'product_owner': ['власник продукту', 'product owner', 'PO'],
        'scrum_master': ['скрам майстер', 'scrum master', 'SM'],
        'team': ['команда', 'team'],
        'planning': ['планування', 'planning'],
        'review': ['огляд', 'review', 'перегляд'],
        'demo': ['демо', 'демонстрація', 'demo'],
        'epic': ['епік', 'epic', 'велике завдання'],
        'task': ['завдання', 'task', 'задача'],
        'bug': ['баг', 'помилка', 'bug', 'дефект'],
        'feature': ['функція', 'feature', 'функціональність'],
        'commit': ['коміт', 'commit', 'фіксація'],
        'merge': ['мердж', 'merge', 'злиття'],
        'branch': ['гілка', 'branch', 'бранч'],
        'pull_request': ['пул реквест', 'pull request', 'PR', 'запит на злиття']
    }

# ============================================================================
# SECTION 2: DATA PROCESSING AND PARSING
# ============================================================================

class VTTProcessor:
    """Process VTT files with Ukrainian text handling"""
    
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        self.morph = pymorphy3.MorphAnalyzer(lang='uk')
        
    def parse_vtt_file(self, filepath: str) -> List[TranscriptSegment]:
        """Parse VTT file and extract segments"""
        segments = []
        
        try:
            captions = webvtt.read(filepath)
            
            for idx, caption in enumerate(captions):
                # Parse timestamps
                start_time = self._parse_timestamp(caption.start)
                end_time = self._parse_timestamp(caption.end)
                
                # Clean text
                text = self._clean_text(caption.text)
                
                # Create segment
                segment = TranscriptSegment(
                    index=idx,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                )
                segments.append(segment)
                
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
        return segments
    
    def _parse_timestamp(self, timestamp_str: str) -> timedelta:
        """Convert VTT timestamp to timedelta"""
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Ukrainian text"""
        # Remove transcription service watermarks
        text = re.sub(r'\(Transcribed by.*?\)', '', text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Fix common OCR/transcription errors in Ukrainian
        replacements = {
            'і': 'і',  # Latin i to Cyrillic і
            'o': 'о',  # Latin o to Cyrillic о
            'a': 'а',  # Latin a to Cyrillic а
            'e': 'е',  # Latin e to Cyrillic е
        }
        
        for lat, cyr in replacements.items():
            text = text.replace(lat, cyr)
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_session_metadata(self, filepath: str) -> SessionMetadata:
        """Extract metadata from filename and content"""
        filename = Path(filepath).stem
        metadata = SessionMetadata(filename=filename, session_type='unknown')
        
        # Parse filename patterns
        if 'Вступ' in filename:
            metadata.session_type = 'introduction'
        elif 'Спринт' in filename:
            metadata.session_type = 'sprint'
            # Extract sprint number
            sprint_match = re.search(r'Спринт_(\d+)', filename)
            if sprint_match:
                metadata.sprint_number = int(sprint_match.group(1))
            # Extract part number
            part_match = re.search(r'частина_(\d+)', filename)
            if part_match:
                metadata.part_number = int(part_match.group(1))
        elif 'Стендап' in filename:
            metadata.session_type = 'standup'
            # Extract standup number
            standup_match = re.search(r'Стендап_(\d+)', filename)
            if standup_match:
                metadata.sprint_number = (int(standup_match.group(1)) - 1) // 3 + 1
        
        return metadata

# ============================================================================
# SECTION 3: SPEAKER DIARIZATION
# ============================================================================

class SpeakerDiarization:
    """Speaker identification and diarization for educational transcripts"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.teacher_patterns = [
            r'(викладач|вчитель|професор|instructor)',
            r'(пояснюю|розглядаємо|завдання|домашнє)',
            r'(лекція|заняття|матеріал)'
        ]
        self.student_patterns = [
            r'(питання|не зрозумів|не розумію|можна)',
            r'(як це|чому|допоможіть|підкажіть)',
            r'(зробив|виконав|спробував)'
        ]
        
    def identify_speakers(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Identify speakers using patterns and clustering"""
        
        # Extract features for each segment
        features = []
        for segment in segments:
            # Linguistic features
            is_teacher = self._is_teacher_utterance(segment.text)
            is_student = self._is_student_utterance(segment.text)
            
            # Sentence embedding
            embedding = self.sentence_model.encode(segment.text)
            
            # Combine features
            feature_vector = np.concatenate([
                [is_teacher, is_student],
                embedding
            ])
            features.append(feature_vector)
        
        # Cluster segments
        features_array = np.array(features)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_array)
        
        # Assign speakers
        for idx, segment in enumerate(segments):
            cluster_id = clustering.labels_[idx]
            if cluster_id == -1:  # Noise point
                segment.speaker = 'Unknown'
            else:
                # Determine if cluster is teacher or student based on patterns
                cluster_segments = [s for i, s in enumerate(segments) 
                                  if clustering.labels_[i] == cluster_id]
                teacher_score = sum(self._is_teacher_utterance(s.text) 
                                  for s in cluster_segments)
                student_score = sum(self._is_student_utterance(s.text) 
                                  for s in cluster_segments)
                
                if teacher_score > student_score:
                    segment.speaker = 'Teacher'
                else:
                    segment.speaker = f'Student_{cluster_id}'
        
        return segments
    
    def _is_teacher_utterance(self, text: str) -> float:
        """Score utterance as likely from teacher"""
        text_lower = text.lower()
        score = 0.0
        for pattern in self.teacher_patterns:
            if re.search(pattern, text_lower):
                score += 1.0
        return score / len(self.teacher_patterns)
    
    def _is_student_utterance(self, text: str) -> float:
        """Score utterance as likely from student"""
        text_lower = text.lower()
        score = 0.0
        for pattern in self.student_patterns:
            if re.search(pattern, text_lower):
                score += 1.0
        return score / len(self.student_patterns)

# ============================================================================
# SECTION 4: ENGAGEMENT ANALYSIS
# ============================================================================

class EngagementAnalyzer:
    """Analyze student engagement patterns"""
    
    def __init__(self):
        self.engagement_indicators = {
            'questions': [
                r'\?', r'чому', r'як', r'коли', r'де', r'що',
                r'можна', r'чи можу', r'підкажіть', r'поясніть'
            ],
            'active_participation': [
                r'я думаю', r'на мою думку', r'мені здається',
                r'я зробив', r'я спробував', r'у мене є'
            ],
            'confusion': [
                r'не зрозумів', r'не розумію', r'складно',
                r'важко', r'заплутано', r'незрозуміло'
            ],
            'understanding': [
                r'зрозуміло', r'ясно', r'добре', r'так',
                r'звичайно', r'логічно'
            ],
            'technical_discussion': [
                r'код', r'функція', r'метод', r'клас',
                r'база даних', r'API', r'фреймворк'
            ]
        }
        
    def calculate_engagement_scores(self, segments: List[TranscriptSegment]) -> pd.DataFrame:
        """Calculate engagement metrics for each segment"""
        engagement_data = []
        
        for segment in segments:
            text_lower = segment.text.lower()
            scores = {}
            
            # Calculate indicator scores
            for indicator, patterns in self.engagement_indicators.items():
                score = sum(1 for pattern in patterns 
                          if re.search(pattern, text_lower))
                scores[indicator] = score / len(patterns)
            
            # Overall engagement score
            scores['overall'] = np.mean(list(scores.values()))
            scores['segment_index'] = segment.index
            scores['timestamp'] = segment.start_time.total_seconds()
            scores['speaker'] = segment.speaker
            
            engagement_data.append(scores)
        
        return pd.DataFrame(engagement_data)
    
    def analyze_participation_frequency(self, segments: List[TranscriptSegment]) -> Dict:
        """Analyze participation frequency by speaker"""
        speaker_stats = defaultdict(lambda: {
            'utterance_count': 0,
            'total_words': 0,
            'question_count': 0,
            'avg_utterance_length': 0,
            'time_spoken': 0
        })
        
        for segment in segments:
            if segment.speaker:
                speaker = segment.speaker
                speaker_stats[speaker]['utterance_count'] += 1
                speaker_stats[speaker]['total_words'] += len(segment.text.split())
                
                # Count questions
                if '?' in segment.text:
                    speaker_stats[speaker]['question_count'] += 1
                
                # Calculate time spoken
                duration = (segment.end_time - segment.start_time).total_seconds()
                speaker_stats[speaker]['time_spoken'] += duration
        
        # Calculate averages
        for speaker in speaker_stats:
            stats = speaker_stats[speaker]
            if stats['utterance_count'] > 0:
                stats['avg_utterance_length'] = stats['total_words'] / stats['utterance_count']
        
        return dict(speaker_stats)
    
    def track_engagement_evolution(self, sessions: List[Dict]) -> pd.DataFrame:
        """Track how engagement evolves across sessions"""
        evolution_data = []
        
        for session in sessions:
            session_engagement = {
                'session_type': session['metadata'].session_type,
                'sprint_number': session['metadata'].sprint_number,
                'avg_questions_per_student': 0,
                'participation_rate': 0,
                'confusion_rate': 0,
                'understanding_rate': 0
            }
            
            # Calculate metrics from engagement scores
            engagement_df = session['engagement_scores']
            student_segments = engagement_df[engagement_df['speaker'].str.contains('Student')]
            
            if len(student_segments) > 0:
                session_engagement['avg_questions_per_student'] = student_segments['questions'].mean()
                session_engagement['participation_rate'] = student_segments['active_participation'].mean()
                session_engagement['confusion_rate'] = student_segments['confusion'].mean()
                session_engagement['understanding_rate'] = student_segments['understanding'].mean()
            
            evolution_data.append(session_engagement)
        
        return pd.DataFrame(evolution_data)

# ============================================================================
# SECTION 5: AGILE ADOPTION ANALYSIS
# ============================================================================

class AgileAdoptionAnalyzer:
    """Analyze adoption of agile concepts and terminology"""
    
    def __init__(self):
        self.agile_terms = AgileTerminology.terms
        self.concept_understanding_patterns = {
            'correct_usage': {
                'sprint': [r'спринт.*тиждень', r'спринт.*планування'],
                'standup': [r'стендап.*щодня', r'стендап.*15 хвилин'],
                'retrospective': [r'ретроспектива.*покращення', r'ретро.*обговорення']
            },
            'misconceptions': {
                'sprint': [r'спринт.*місяць', r'спринт.*довго'],
                'standup': [r'стендап.*година', r'стендап.*звіт'],
                'retrospective': [r'ретро.*критика', r'ретро.*помилки']
            }
        }
        
    def analyze_terminology_usage(self, segments: List[TranscriptSegment]) -> Dict:
        """Analyze agile terminology usage patterns"""
        term_usage = defaultdict(lambda: {
            'frequency': 0,
            'speakers': set(),
            'contexts': [],
            'correct_usage': 0,
            'misconceptions': 0
        })
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            for term_key, term_variations in self.agile_terms.items():
                for variation in term_variations:
                    if variation in text_lower:
                        term_usage[term_key]['frequency'] += 1
                        term_usage[term_key]['speakers'].add(segment.speaker)
                        term_usage[term_key]['contexts'].append({
                            'text': segment.text,
                            'speaker': segment.speaker,
                            'timestamp': segment.start_time
                        })
                        
                        # Check for correct usage
                        if term_key in self.concept_understanding_patterns['correct_usage']:
                            for pattern in self.concept_understanding_patterns['correct_usage'][term_key]:
                                if re.search(pattern, text_lower):
                                    term_usage[term_key]['correct_usage'] += 1
                        
                        # Check for misconceptions
                        if term_key in self.concept_understanding_patterns['misconceptions']:
                            for pattern in self.concept_understanding_patterns['misconceptions'][term_key]:
                                if re.search(pattern, text_lower):
                                    term_usage[term_key]['misconceptions'] += 1
        
        # Convert sets to lists for JSON serialization
        for term in term_usage:
            term_usage[term]['speakers'] = list(term_usage[term]['speakers'])
        
        return dict(term_usage)
    
    def calculate_adoption_metrics(self, sessions: List[Dict]) -> pd.DataFrame:
        """Calculate agile adoption metrics across sessions"""
        adoption_data = []
        
        for session in sessions:
            term_usage = session['agile_terminology']
            
            metrics = {
                'session': session['metadata'].filename,
                'session_type': session['metadata'].session_type,
                'sprint_number': session['metadata'].sprint_number,
                'total_agile_terms': sum(usage['frequency'] for usage in term_usage.values()),
                'unique_terms_used': len([t for t, u in term_usage.items() if u['frequency'] > 0]),
                'correct_usage_rate': 0,
                'misconception_rate': 0,
                'student_adoption_rate': 0
            }
            
            # Calculate rates
            total_usage = sum(u['frequency'] for u in term_usage.values())
            total_correct = sum(u['correct_usage'] for u in term_usage.values())
            total_misconceptions = sum(u['misconceptions'] for u in term_usage.values())
            
            if total_usage > 0:
                metrics['correct_usage_rate'] = total_correct / total_usage
                metrics['misconception_rate'] = total_misconceptions / total_usage
            
            # Calculate student adoption
            student_usage = sum(1 for u in term_usage.values() 
                              for s in u['speakers'] if 'Student' in str(s))
            total_speakers = sum(len(u['speakers']) for u in term_usage.values())
            
            if total_speakers > 0:
                metrics['student_adoption_rate'] = student_usage / total_speakers
            
            adoption_data.append(metrics)
        
        return pd.DataFrame(adoption_data)

# ============================================================================
# SECTION 6: PROBLEM IDENTIFICATION
# ============================================================================

class ProblemIdentifier:
    """Identify technical and conceptual problems in discussions"""
    
    def __init__(self):
        self.problem_indicators = {
            'technical_issues': [
                r'помилка', r'error', r'не працює', r'зламалось',
                r'баг', r'bug', r'exception', r'падає', r'креш'
            ],
            'conceptual_difficulties': [
                r'не розумію', r'не зрозуміло', r'заплутано',
                r'складно', r'важко', r'не можу зрозуміти'
            ],
            'process_challenges': [
                r'не встигаю', r'забагато', r'немає часу',
                r'не знаю з чого почати', r'заблокований'
            ],
            'collaboration_issues': [
                r'конфлікт', r'не домовились', r'різні думки',
                r'не можемо знайти', r'команда не'
            ],
            'tool_problems': [
                r'не встановлюється', r'не запускається',
                r'проблема з git', r'IDE не', r'npm error'
            ]
        }
        
    def identify_problems(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """Identify and categorize problems mentioned in transcripts"""
        problems = []
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            for category, indicators in self.problem_indicators.items():
                for indicator in indicators:
                    if re.search(indicator, text_lower):
                        problem = {
                            'category': category,
                            'indicator': indicator,
                            'text': segment.text,
                            'speaker': segment.speaker,
                            'timestamp': segment.start_time.total_seconds(),
                            'context_before': None,
                            'context_after': None
                        }
                        
                        # Add context (previous and next segments)
                        if segment.index > 0:
                            problem['context_before'] = segments[segment.index - 1].text
                        if segment.index < len(segments) - 1:
                            problem['context_after'] = segments[segment.index + 1].text
                        
                        problems.append(problem)
        
        return problems
    
    def analyze_problem_patterns(self, sessions: List[Dict]) -> pd.DataFrame:
        """Analyze patterns in problems across sessions"""
        problem_patterns = []
        
        for session in sessions:
            problems = session['problems']
            
            # Count problems by category
            category_counts = Counter(p['category'] for p in problems)
            
            pattern = {
                'session': session['metadata'].filename,
                'session_type': session['metadata'].session_type,
                'sprint_number': session['metadata'].sprint_number,
                'total_problems': len(problems),
                **{f'{cat}_count': count for cat, count in category_counts.items()},
                'student_problems': sum(1 for p in problems if 'Student' in str(p['speaker'])),
                'teacher_problems': sum(1 for p in problems if p['speaker'] == 'Teacher'),
                'avg_problem_time': np.mean([p['timestamp'] for p in problems]) if problems else 0
            }
            
            problem_patterns.append(pattern)
        
        return pd.DataFrame(problem_patterns)
    
    def find_recurring_problems(self, sessions: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
        """Find problems that recur across sessions"""
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        all_problems = []
        for session in sessions:
            for problem in session['problems']:
                problem['session'] = session['metadata'].filename
                all_problems.append(problem)
        
        # Create embeddings for all problem texts
        problem_texts = [p['text'] for p in all_problems]
        embeddings = model.encode(problem_texts)
        
        # Find similar problems
        recurring_problems = []
        processed = set()
        
        for i, prob1 in enumerate(all_problems):
            if i in processed:
                continue
            
            similar_problems = [prob1]
            processed.add(i)
            
            for j, prob2 in enumerate(all_problems[i+1:], i+1):
                if j in processed:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity > similarity_threshold:
                    similar_problems.append(prob2)
                    processed.add(j)
            
            if len(similar_problems) > 1:
                recurring_problems.append({
                    'problem_cluster': similar_problems[0]['text'][:100] + '...',
                    'category': similar_problems[0]['category'],
                    'occurrences': len(similar_problems),
                    'sessions': list(set(p['session'] for p in similar_problems)),
                    'examples': similar_problems[:3]  # Keep first 3 examples
                })
        
        return recurring_problems

# ============================================================================
# SECTION 7: SENTIMENT AND TOPIC ANALYSIS
# ============================================================================

class SentimentTopicAnalyzer:
    """Sentiment analysis and topic modeling for Ukrainian text"""
    
    def __init__(self):
        # Initialize Ukrainian sentiment analyzer
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="youscan/ukr-roberta-sentiment"
            )
        except:
            print("Ukrainian sentiment model not available, using multilingual fallback")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        
    def analyze_sentiment(self, segments: List[TranscriptSegment]) -> pd.DataFrame:
        """Analyze sentiment for each segment"""
        sentiment_data = []
        
        for segment in segments:
            # Get sentiment
            try:
                result = self.sentiment_pipeline(segment.text[:512])[0]  # Limit length
                sentiment = {
                    'segment_index': segment.index,
                    'speaker': segment.speaker,
                    'sentiment_label': result['label'],
                    'sentiment_score': result['score'],
                    'timestamp': segment.start_time.total_seconds()
                }
            except:
                sentiment = {
                    'segment_index': segment.index,
                    'speaker': segment.speaker,
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.5,
                    'timestamp': segment.start_time.total_seconds()
                }
            
            sentiment_data.append(sentiment)
        
        return pd.DataFrame(sentiment_data)
    
    def extract_topics(self, segments: List[TranscriptSegment], n_topics: int = 10) -> Dict:
        """Extract topics using LDA"""
        # Prepare texts
        texts = [segment.text for segment in segments]
        
        # Ukrainian stop words
        stop_words = [
            'і', 'в', 'на', 'з', 'до', 'що', 'це', 'як', 'для', 'від',
            'про', 'так', 'але', 'чи', 'у', 'я', 'ви', 'він', 'вона', 'ми',
            'вони', 'бути', 'мати', 'робити', 'казати', 'той', 'цей', 'весь',
            'який', 'коли', 'де', 'тут', 'там', 'зараз', 'дуже', 'можна', 'треба'
        ]
        
        # Vectorize
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words=stop_words,
            ngram_range=(1, 2)
        )
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online'
        )
        lda.fit(doc_term_matrix)
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': topic[top_indices].tolist()
            })
        
        # Assign topics to segments
        doc_topics = lda.transform(doc_term_matrix)
        for idx, segment in enumerate(segments):
            segment.topics = doc_topics[idx].argsort()[-3:][::-1].tolist()
        
        return {
            'topics': topics,
            'doc_topic_distribution': doc_topics.tolist()
        }
    
    def analyze_topic_evolution(self, sessions: List[Dict]) -> pd.DataFrame:
        """Analyze how topics evolve across sessions"""
        evolution_data = []
        
        for session in sessions:
            topics = session['topics']
            
            # Calculate topic distribution
            doc_topic_dist = np.array(topics['doc_topic_distribution'])
            avg_topic_dist = doc_topic_dist.mean(axis=0)
            
            topic_evolution = {
                'session': session['metadata'].filename,
                'session_type': session['metadata'].session_type,
                'sprint_number': session['metadata'].sprint_number,
                'dominant_topic': int(avg_topic_dist.argmax()),
                'topic_diversity': float(np.std(avg_topic_dist)),
                **{f'topic_{i}_weight': float(w) for i, w in enumerate(avg_topic_dist)}
            }
            
            evolution_data.append(topic_evolution)
        
        return pd.DataFrame(evolution_data)

# ============================================================================
# SECTION 8: TEACHING EFFECTIVENESS ANALYSIS
# ============================================================================

class TeachingEffectivenessAnalyzer:
    """Analyze teaching effectiveness markers"""
    
    def __init__(self):
        self.effectiveness_indicators = {
            'clear_explanation': [
                r'наприклад', r'тобто', r'іншими словами',
                r'давайте розглянемо', r'покажу', r'демонструю'
            ],
            'checking_understanding': [
                r'зрозуміло\?', r'питання\?', r'все ясно\?',
                r'хто має питання', r'давайте перевіримо'
            ],
            'encouragement': [
                r'молодець', r'добре', r'чудово', r'правильно',
                r'гарна робота', r'так тримати'
            ],
            'scaffolding': [
                r'спочатку', r'потім', r'крок за кроком',
                r'почнемо з', r'далі', r'наступний крок'
            ],
            'real_world_connection': [
                r'на практиці', r'в реальному проекті',
                r'у компанії', r'досвід показує'
            ]
        }
        
    def analyze_teaching_patterns(self, segments: List[TranscriptSegment]) -> Dict:
        """Analyze teaching effectiveness patterns"""
        teacher_segments = [s for s in segments if s.speaker == 'Teacher']
        
        effectiveness_scores = defaultdict(list)
        
        for segment in teacher_segments:
            text_lower = segment.text.lower()
            
            for indicator, patterns in self.effectiveness_indicators.items():
                score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
                effectiveness_scores[indicator].append(score / len(patterns))
        
        # Calculate statistics
        stats = {}
        for indicator, scores in effectiveness_scores.items():
            stats[indicator] = {
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'frequency': sum(1 for s in scores if s > 0) / len(teacher_segments) if teacher_segments else 0
            }
        
        return stats
    
    def analyze_interaction_patterns(self, segments: List[TranscriptSegment]) -> Dict:
        """Analyze teacher-student interaction patterns"""
        interactions = []
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            # Identify interaction types
            if current.speaker == 'Teacher' and next_seg.speaker and 'Student' in next_seg.speaker:
                interaction_type = 'teacher_to_student'
            elif current.speaker and 'Student' in current.speaker and next_seg.speaker == 'Teacher':
                interaction_type = 'student_to_teacher'
            else:
                continue
            
            interactions.append({
                'type': interaction_type,
                'timestamp': current.start_time.total_seconds(),
                'teacher_text': current.text if current.speaker == 'Teacher' else next_seg.text,
                'student_text': next_seg.text if 'Student' in str(next_seg.speaker) else current.text
            })
        
        # Calculate interaction metrics
        metrics = {
            'total_interactions': len(interactions),
            'teacher_initiated': sum(1 for i in interactions if i['type'] == 'teacher_to_student'),
            'student_initiated': sum(1 for i in interactions if i['type'] == 'student_to_teacher'),
            'avg_interaction_interval': np.mean(np.diff([i['timestamp'] for i in interactions])) if len(interactions) > 1 else 0
        }
        
        return metrics

# ============================================================================
# SECTION 9: STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Statistical analysis for educational research"""
    
    def __init__(self):
        pass
    
    def compare_sprints(self, sprint_data: pd.DataFrame, metric: str) -> Dict:
        """Compare metrics across sprints using appropriate statistical tests"""
        results = {}
        
        # Kruskal-Wallis test for multiple groups
        sprint_groups = [group[metric].values for _, group in sprint_data.groupby('sprint_number')]
        
        if len(sprint_groups) > 2:
            h_stat, p_value = kruskal(*sprint_groups)
            results['kruskal_wallis'] = {
                'statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Post-hoc pairwise comparisons if significant
            if p_value < 0.05:
                pairwise_results = []
                for i in range(len(sprint_groups)):
                    for j in range(i+1, len(sprint_groups)):
                        u_stat, p_val = mannwhitneyu(sprint_groups[i], sprint_groups[j])
                        pairwise_results.append({
                            'sprint_1': i+1,
                            'sprint_2': j+1,
                            'u_statistic': u_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05/len(sprint_groups)  # Bonferroni correction
                        })
                results['pairwise_comparisons'] = pairwise_results
        
        # Effect size (Cohen's d)
        if len(sprint_groups) == 2:
            d = (np.mean(sprint_groups[1]) - np.mean(sprint_groups[0])) / np.sqrt(
                (np.var(sprint_groups[0]) + np.var(sprint_groups[1])) / 2
            )
            results['effect_size'] = d
        
        return results
    
    def analyze_correlation(self, data: pd.DataFrame, var1: str, var2: str) -> Dict:
        """Analyze correlation between variables"""
        # Remove NaN values
        clean_data = data[[var1, var2]].dropna()
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(clean_data[var1], clean_data[var2])
        
        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(clean_data[var1], clean_data[var2])
        
        return {
            'pearson': {'r': pearson_r, 'p_value': pearson_p},
            'spearman': {'rho': spearman_r, 'p_value': spearman_p},
            'n_observations': len(clean_data)
        }
    
    def time_series_analysis(self, data: pd.DataFrame, metric: str) -> Dict:
        """Analyze trends over time"""
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        
        # Linear regression for trend
        X = np.arange(len(data_sorted)).reshape(-1, 1)
        y = data_sorted[metric].values
        
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        return {
            'trend_coefficient': model.params[1],
            'r_squared': model.rsquared,
            'p_value': model.pvalues[1],
            'confidence_interval': model.conf_int()[1].tolist(),
            'trend_direction': 'increasing' if model.params[1] > 0 else 'decreasing'
        }

# ============================================================================
# SECTION 10: VISUALIZATION
# ============================================================================

class ResearchVisualizer:
    """Create visualizations for educational research papers"""
    
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)
        
    def plot_engagement_evolution(self, evolution_data: pd.DataFrame, output_path: str):
        """Plot engagement metrics evolution across sprints"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['avg_questions_per_student', 'participation_rate', 
                  'confusion_rate', 'understanding_rate']
        titles = ['Questions per Student', 'Participation Rate', 
                 'Confusion Rate', 'Understanding Rate']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            sprint_data = evolution_data[evolution_data['sprint_number'].notna()]
            
            ax.plot(sprint_data['sprint_number'], sprint_data[metric], 
                   marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Sprint Number')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Across Sprints')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_agile_adoption(self, adoption_data: pd.DataFrame, output_path: str):
        """Plot agile terminology adoption patterns"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Adoption rate over time
        sprint_data = adoption_data[adoption_data['sprint_number'].notna()]
        ax1.plot(sprint_data['sprint_number'], sprint_data['student_adoption_rate'], 
                marker='s', label='Student Adoption', linewidth=2)
        ax1.plot(sprint_data['sprint_number'], sprint_data['correct_usage_rate'], 
                marker='^', label='Correct Usage', linewidth=2)
        ax1.set_xlabel('Sprint Number')
        ax1.set_ylabel('Rate')
        ax1.set_title('Agile Terminology Adoption Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Term frequency by session type
        session_types = adoption_data.groupby('session_type')['total_agile_terms'].mean()
        ax2.bar(session_types.index, session_types.values, color=self.colors[:len(session_types)])
        ax2.set_xlabel('Session Type')
        ax2.set_ylabel('Average Agile Terms Used')
        ax2.set_title('Agile Terminology Usage by Session Type')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_problem_heatmap(self, problem_patterns: pd.DataFrame, output_path: str):
        """Create heatmap of problems across sessions"""
        problem_categories = ['technical_issues_count', 'conceptual_difficulties_count',
                            'process_challenges_count', 'collaboration_issues_count']
        
        # Prepare data for heatmap
        heatmap_data = problem_patterns[['session', 'sprint_number'] + problem_categories]
        heatmap_data = heatmap_data.set_index('session')[problem_categories].T
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', 
                   cbar_kws={'label': 'Problem Count'})
        plt.title('Problem Distribution Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('Problem Category')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_speaker_network(self, interactions: List[Dict], output_path: str):
        """Create network visualization of speaker interactions"""
        G = nx.DiGraph()
        
        # Count interactions between speakers
        interaction_counts = defaultdict(int)
        for interaction in interactions:
            if interaction['type'] == 'teacher_to_student':
                interaction_counts[('Teacher', 'Students')] += 1
            else:
                interaction_counts[('Students', 'Teacher')] += 1
        
        # Add edges with weights
        for (source, target), count in interaction_counts.items():
            G.add_edge(source, target, weight=count)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=self.colors[:len(G.nodes())])
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w/10 for w in weights], alpha=0.6, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title('Teacher-Student Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_wordcloud_by_sprint(self, sessions: List[Dict], output_path: str):
        """Create word clouds for each sprint"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for sprint_num in range(1, 4):
            # Collect text from sprint sessions
            sprint_text = ""
            for session in sessions:
                if session['metadata'].sprint_number == sprint_num:
                    for segment in session['segments']:
                        sprint_text += " " + segment.text
            
            # Create word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis'
            ).generate(sprint_text)
            
            axes[sprint_num-1].imshow(wordcloud, interpolation='bilinear')
            axes[sprint_num-1].set_title(f'Sprint {sprint_num} Word Cloud')
            axes[sprint_num-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# SECTION 11: MAIN ANALYSIS PIPELINE
# ============================================================================

class AgileEducationAnalyzer:
    """Main analysis pipeline for agile education research"""
    
    def __init__(self, transcripts_dir: str):
        self.transcripts_dir = Path(transcripts_dir)
        self.vtt_processor = VTTProcessor()
        self.speaker_diarizer = SpeakerDiarization()
        self.engagement_analyzer = EngagementAnalyzer()
        self.agile_analyzer = AgileAdoptionAnalyzer()
        self.problem_identifier = ProblemIdentifier()
        self.sentiment_analyzer = SentimentTopicAnalyzer()
        self.teaching_analyzer = TeachingEffectivenessAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResearchVisualizer()
        
    def analyze_all_sessions(self) -> Dict:
        """Run complete analysis on all sessions"""
        sessions = []
        
        # Process each VTT file
        for vtt_file in sorted(self.transcripts_dir.glob('*.vtt')):
            print(f"Processing {vtt_file.name}...")
            
            # Parse VTT
            segments = self.vtt_processor.parse_vtt_file(str(vtt_file))
            
            # Extract metadata
            metadata = self.vtt_processor.extract_session_metadata(str(vtt_file))
            
            # Speaker diarization
            segments = self.speaker_diarizer.identify_speakers(segments)
            
            # Engagement analysis
            engagement_scores = self.engagement_analyzer.calculate_engagement_scores(segments)
            participation_stats = self.engagement_analyzer.analyze_participation_frequency(segments)
            
            # Agile terminology analysis
            terminology_usage = self.agile_analyzer.analyze_terminology_usage(segments)
            
            # Problem identification
            problems = self.problem_identifier.identify_problems(segments)
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.analyze_sentiment(segments)
            
            # Topic modeling
            topics = self.sentiment_analyzer.extract_topics(segments)
            
            # Teaching effectiveness
            teaching_patterns = self.teaching_analyzer.analyze_teaching_patterns(segments)
            interaction_patterns = self.teaching_analyzer.analyze_interaction_patterns(segments)
            
            # Store session data
            sessions.append({
                'metadata': metadata,
                'segments': segments,
                'engagement_scores': engagement_scores,
                'participation_stats': participation_stats,
                'agile_terminology': terminology_usage,
                'problems': problems,
                'sentiment_scores': sentiment_scores,
                'topics': topics,
                'teaching_patterns': teaching_patterns,
                'interaction_patterns': interaction_patterns
            })
        
        # Cross-session analyses
        print("Performing cross-session analyses...")
        
        # Engagement evolution
        engagement_evolution = self.engagement_analyzer.track_engagement_evolution(sessions)
        
        # Agile adoption metrics
        adoption_metrics = self.agile_analyzer.calculate_adoption_metrics(sessions)
        
        # Problem patterns
        problem_patterns = self.problem_identifier.analyze_problem_patterns(sessions)
        recurring_problems = self.problem_identifier.find_recurring_problems(sessions)
        
        # Topic evolution
        topic_evolution = self.sentiment_analyzer.analyze_topic_evolution(sessions)
        
        # Statistical analyses
        statistical_results = {}
        if not engagement_evolution.empty and 'sprint_number' in engagement_evolution.columns:
            statistical_results['participation_comparison'] = self.statistical_analyzer.compare_sprints(
                engagement_evolution[engagement_evolution['sprint_number'].notna()], 
                'participation_rate'
            )
        
        return {
            'sessions': sessions,
            'engagement_evolution': engagement_evolution,
            'adoption_metrics': adoption_metrics,
            'problem_patterns': problem_patterns,
            'recurring_problems': recurring_problems,
            'topic_evolution': topic_evolution,
            'statistical_results': statistical_results
        }
    
    def generate_visualizations(self, analysis_results: Dict, output_dir: str):
        """Generate all visualizations for research paper"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating visualizations...")
        
        # Engagement evolution plot
        if not analysis_results['engagement_evolution'].empty:
            self.visualizer.plot_engagement_evolution(
                analysis_results['engagement_evolution'],
                str(output_path / 'engagement_evolution.png')
            )
        
        # Agile adoption plot
        if not analysis_results['adoption_metrics'].empty:
            self.visualizer.plot_agile_adoption(
                analysis_results['adoption_metrics'],
                str(output_path / 'agile_adoption.png')
            )
        
        # Problem heatmap
        if not analysis_results['problem_patterns'].empty:
            self.visualizer.create_problem_heatmap(
                analysis_results['problem_patterns'],
                str(output_path / 'problem_heatmap.png')
            )
        
        # Word clouds by sprint
        self.visualizer.create_wordcloud_by_sprint(
            analysis_results['sessions'],
            str(output_path / 'sprint_wordclouds.png')
        )
        
    def generate_report(self, analysis_results: Dict, output_file: str):
        """Generate comprehensive research report"""
        report = []
        report.append("# Agile Education Research Analysis Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"- Total Sessions Analyzed: {len(analysis_results['sessions'])}\n")
        report.append(f"- Total Problems Identified: {analysis_results['problem_patterns']['total_problems'].sum()}\n")
        report.append(f"- Recurring Problems: {len(analysis_results['recurring_problems'])}\n\n")
        
        # Key Findings
        report.append("## Key Findings\n\n")
        
        # Engagement trends
        report.append("### Student Engagement\n")
        evolution = analysis_results['engagement_evolution']
        if not evolution.empty and 'sprint_number' in evolution.columns:
            sprint_data = evolution[evolution['sprint_number'].notna()]
            if not sprint_data.empty:
                report.append(f"- Average participation rate: {sprint_data['participation_rate'].mean():.2%}\n")
                report.append(f"- Participation trend: {'Increasing' if sprint_data['participation_rate'].iloc[-1] > sprint_data['participation_rate'].iloc[0] else 'Decreasing'}\n")
        
        # Agile adoption
        report.append("\n### Agile Concept Adoption\n")
        adoption = analysis_results['adoption_metrics']
        if not adoption.empty:
            report.append(f"- Average correct usage rate: {adoption['correct_usage_rate'].mean():.2%}\n")
            report.append(f"- Student adoption rate: {adoption['student_adoption_rate'].mean():.2%}\n")
        
        # Common problems
        report.append("\n### Common Challenges\n")
        for problem in analysis_results['recurring_problems'][:5]:
            report.append(f"- {problem['category']}: {problem['occurrences']} occurrences across {len(problem['sessions'])} sessions\n")
        
        # Statistical significance
        report.append("\n### Statistical Results\n")
        for test_name, results in analysis_results['statistical_results'].items():
            if 'p_value' in results:
                report.append(f"- {test_name}: p-value = {results['p_value']:.4f} ")
                report.append(f"({'Significant' if results.get('significant', False) else 'Not significant'})\n")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"Report saved to {output_file}")

# ============================================================================
# SECTION 12: USAGE EXAMPLE
# ============================================================================

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = AgileEducationAnalyzer('/mnt/project')
    
    # Run analysis
    print("Starting comprehensive analysis...")
    results = analyzer.analyze_all_sessions()
    
    # Generate visualizations
    analyzer.generate_visualizations(results, '/mnt/user-data/outputs/visualizations')
    
    # Generate report
    analyzer.generate_report(results, '/mnt/user-data/outputs/analysis_report.md')
    
    print("Analysis complete!")
    
    # Research questions answers
    print("\n=== Research Questions Answers ===\n")
    
    # Q1: How does student participation evolve across sprints?
    evolution = results['engagement_evolution']
    if not evolution.empty and 'sprint_number' in evolution.columns:
        sprint_data = evolution[evolution['sprint_number'].notna()]
        if not sprint_data.empty:
            participation_change = (sprint_data['participation_rate'].iloc[-1] - 
                                   sprint_data['participation_rate'].iloc[0]) / sprint_data['participation_rate'].iloc[0] * 100
            print(f"Q1: Student participation changed by {participation_change:.1f}% from Sprint 1 to Sprint 3")
    
    # Q2: What agile concepts are most/least understood?
    adoption = results['adoption_metrics']
    if not adoption.empty:
        print(f"Q2: Overall correct usage rate of agile concepts: {adoption['correct_usage_rate'].mean():.1%}")
        print(f"    Misconception rate: {adoption['misconception_rate'].mean():.1%}")
    
    # Q3: What technical challenges emerge repeatedly?
    print(f"Q3: Found {len(results['recurring_problems'])} recurring problems")
    if results['recurring_problems']:
        most_common = results['recurring_problems'][0]
        print(f"    Most common: {most_common['category']} ({most_common['occurrences']} times)")
    
    # Q4: How effective are stand-ups for student learning?
    standup_sessions = [s for s in results['sessions'] if s['metadata'].session_type == 'standup']
    if standup_sessions:
        avg_standup_engagement = np.mean([s['engagement_scores']['overall'].mean() 
                                         for s in standup_sessions])
        print(f"Q4: Average engagement in stand-ups: {avg_standup_engagement:.2f}")

if __name__ == "__main__":
    main()
