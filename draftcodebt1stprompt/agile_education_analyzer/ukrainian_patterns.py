"""
Ukrainian Discourse Pattern Detection Module
Specialized pattern matching for Ukrainian educational transcripts
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .data_structures import (
    TranscriptSegment,
    UkrainianDiscoursePattern,
    DiscourseType
)
from .utils.logger import get_logger

logger = get_logger('ukrainian_patterns')

class UkrainianDiscourseDetector:
    """
    Detect and classify Ukrainian educational discourse patterns.

    This class handles Ukrainian-specific linguistic patterns including:
    - Question phrases
    - Confusion indicators
    - Understanding confirmations
    - Teacher explanations
    - Code-switching between Ukrainian and English
    """

    def __init__(self):
        """Initialize with Ukrainian discourse patterns"""
        self.patterns = UkrainianDiscoursePattern()
        logger.info("Ukrainian discourse detector initialized")

    def detect_questions(self, text: str) -> Dict:
        """
        Detect if text contains a question and classify question type.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with is_question flag and question type
        """
        text_lower = text.lower().strip()

        result = {
            'is_question': False,
            'question_type': None,
            'confidence': 0.0,
            'matched_patterns': []
        }

        # Check each question category
        for category, patterns in self.patterns.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    result['is_question'] = True
                    result['question_type'] = category
                    result['confidence'] = 1.0 if text_lower.endswith('?') else 0.8
                    result['matched_patterns'].append(pattern)
                    break
            if result['is_question']:
                break

        return result

    def detect_confusion(self, text: str) -> Dict:
        """
        Detect confusion indicators in Ukrainian text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with is_confused flag and confusion indicators
        """
        text_lower = text.lower()

        matched_indicators = []
        for pattern in self.patterns.confusion_patterns:
            if re.search(pattern, text_lower):
                matched_indicators.append(pattern)

        is_confused = len(matched_indicators) > 0

        return {
            'is_confused': is_confused,
            'confusion_level': self._calculate_confusion_level(matched_indicators),
            'matched_indicators': matched_indicators,
            'indicator_count': len(matched_indicators)
        }

    def detect_understanding(self, text: str) -> Dict:
        """
        Detect understanding confirmations.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with understanding indicators
        """
        text_lower = text.lower().strip()

        matched_patterns = []
        for pattern in self.patterns.understanding_patterns:
            if re.search(pattern, text_lower):
                matched_patterns.append(pattern)

        is_understanding = len(matched_patterns) > 0

        # Higher confidence for exact matches
        exact_matches = [
            'зрозуміло', 'ясно', 'так', 'добре', 'окей', 'ok'
        ]
        confidence = 1.0 if text_lower in exact_matches else 0.7

        return {
            'is_understanding': is_understanding,
            'confidence': confidence if is_understanding else 0.0,
            'matched_patterns': matched_patterns
        }

    def detect_teacher_explanation(self, text: str) -> Dict:
        """
        Detect teacher explanation patterns.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with explanation indicators
        """
        text_lower = text.lower()

        explanation_matches = []
        for pattern in self.patterns.teacher_explanation_patterns:
            if re.search(pattern, text_lower):
                explanation_matches.append(pattern)

        instruction_matches = []
        for pattern in self.patterns.teacher_instruction_patterns:
            if re.search(pattern, text_lower):
                instruction_matches.append(pattern)

        is_explanation = len(explanation_matches) > 0
        is_instruction = len(instruction_matches) > 0

        if is_explanation and is_instruction:
            discourse_type = 'explanation_with_instruction'
        elif is_explanation:
            discourse_type = 'explanation'
        elif is_instruction:
            discourse_type = 'instruction'
        else:
            discourse_type = None

        return {
            'is_teacher_discourse': is_explanation or is_instruction,
            'discourse_type': discourse_type,
            'explanation_indicators': explanation_matches,
            'instruction_indicators': instruction_matches
        }

    def detect_positive_feedback(self, text: str) -> Dict:
        """
        Detect positive feedback patterns.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with feedback indicators
        """
        text_lower = text.lower()

        matched_patterns = []
        for pattern in self.patterns.positive_feedback_patterns:
            if re.search(pattern, text_lower):
                matched_patterns.append(pattern)

        is_positive_feedback = len(matched_patterns) > 0

        return {
            'is_positive_feedback': is_positive_feedback,
            'matched_patterns': matched_patterns,
            'feedback_strength': len(matched_patterns)
        }

    def detect_code_switching(self, text: str) -> Dict:
        """
        Detect code-switching between Ukrainian and English.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with code-switching analysis
        """
        # Count English technical terms
        english_terms = []
        for pattern in self.patterns.code_switching_indicators:
            matches = re.findall(r'\b' + pattern + r'\b', text, re.IGNORECASE)
            english_terms.extend(matches)

        # Detect mixed language sentences (contains both Cyrillic and Latin)
        has_cyrillic = bool(re.search('[а-яА-ЯіїєґІЇЄҐ]', text))
        has_latin = bool(re.search('[a-zA-Z]', text))
        is_mixed = has_cyrillic and has_latin

        # Calculate code-switching ratio
        words = text.split()
        total_words = len(words)
        english_word_count = len(english_terms)
        switching_ratio = english_word_count / total_words if total_words > 0 else 0

        return {
            'has_code_switching': is_mixed,
            'english_terms': english_terms,
            'english_term_count': english_word_count,
            'total_words': total_words,
            'switching_ratio': switching_ratio,
            'switching_intensity': self._categorize_switching_intensity(switching_ratio)
        }

    def analyze_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """
        Comprehensive analysis of a transcript segment.

        Applies all pattern detection methods and updates segment attributes.

        Args:
            segment: Transcript segment to analyze

        Returns:
            Updated segment with pattern detection results
        """
        text = segment.text

        # Question detection
        question_result = self.detect_questions(text)
        segment.is_question = question_result['is_question']

        # Confusion detection
        confusion_result = self.detect_confusion(text)
        segment.is_confusion = confusion_result['is_confused']

        # Understanding detection
        understanding_result = self.detect_understanding(text)
        segment.is_understanding = understanding_result['is_understanding']

        # Code-switching detection
        code_switching = self.detect_code_switching(text)

        # Store technical terms if found
        if code_switching['has_code_switching']:
            segment.technical_terms = code_switching['english_terms']

        return segment

    def classify_discourse_type(self, text: str, speaker_role: Optional[str] = None) -> str:
        """
        Classify the type of educational discourse.

        Args:
            text: Text to classify
            speaker_role: Optional speaker role ('teacher' or 'student')

        Returns:
            Discourse type classification
        """
        question = self.detect_questions(text)
        confusion = self.detect_confusion(text)
        understanding = self.detect_understanding(text)
        teacher_discourse = self.detect_teacher_explanation(text)
        feedback = self.detect_positive_feedback(text)

        # Priority-based classification
        if question['is_question']:
            if confusion['is_confused']:
                return 'confused_question'
            return 'question'

        if confusion['is_confused']:
            return 'confusion'

        if understanding['is_understanding']:
            return 'confirmation'

        if feedback['is_positive_feedback']:
            return 'positive_feedback'

        if teacher_discourse['is_teacher_discourse']:
            return teacher_discourse['discourse_type']

        # Default based on speaker role
        if speaker_role == 'teacher':
            return 'teacher_talk'
        elif speaker_role == 'student':
            return 'student_talk'

        return 'general_discussion'

    def extract_key_phrases(self, segments: List[TranscriptSegment],
                           phrase_type: str = 'questions') -> List[Dict]:
        """
        Extract specific types of phrases from segments.

        Args:
            segments: List of segments
            phrase_type: Type of phrases ('questions', 'confusions', 'confirmations')

        Returns:
            List of extracted phrases with metadata
        """
        extracted = []

        for segment in segments:
            if phrase_type == 'questions' and segment.is_question:
                extracted.append({
                    'text': segment.text,
                    'timestamp': segment.start_time,
                    'speaker': segment.speaker,
                    'segment_index': segment.index
                })
            elif phrase_type == 'confusions' and segment.is_confusion:
                extracted.append({
                    'text': segment.text,
                    'timestamp': segment.start_time,
                    'speaker': segment.speaker,
                    'segment_index': segment.index
                })
            elif phrase_type == 'confirmations' and segment.is_understanding:
                extracted.append({
                    'text': segment.text,
                    'timestamp': segment.start_time,
                    'speaker': segment.speaker,
                    'segment_index': segment.index
                })

        logger.info(f"Extracted {len(extracted)} {phrase_type}")
        return extracted

    def _calculate_confusion_level(self, indicators: List[str]) -> str:
        """Categorize confusion level based on number of indicators"""
        count = len(indicators)
        if count == 0:
            return 'none'
        elif count == 1:
            return 'mild'
        elif count == 2:
            return 'moderate'
        else:
            return 'severe'

    def _categorize_switching_intensity(self, ratio: float) -> str:
        """Categorize code-switching intensity"""
        if ratio < 0.1:
            return 'minimal'
        elif ratio < 0.3:
            return 'moderate'
        elif ratio < 0.5:
            return 'frequent'
        else:
            return 'dominant'

class DialoguePatternAnalyzer:
    """
    Analyze educational dialogue patterns and interactions.

    Detects:
    - Teacher-student question-answer pairs
    - Peer-to-peer interactions
    - Explanation sequences
    - IRE (Initiation-Response-Evaluation) patterns
    """

    def __init__(self):
        """Initialize dialogue analyzer"""
        self.detector = UkrainianDiscourseDetector()
        logger.info("Dialogue pattern analyzer initialized")

    def detect_qa_pairs(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """
        Detect question-answer pairs in dialogue.

        Args:
            segments: List of transcript segments

        Returns:
            List of detected Q&A pairs with metadata
        """
        qa_pairs = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # If current segment is a question
            if current.is_question:
                # Check if next segment is from different speaker (likely answer)
                if (next_seg.speaker and current.speaker and
                    next_seg.speaker != current.speaker):

                    # Time gap should be reasonable (< 10 seconds)
                    time_gap = (next_seg.start_time - current.end_time).total_seconds()
                    if time_gap < 10:
                        qa_pairs.append({
                            'question': current.text,
                            'question_speaker': current.speaker,
                            'question_timestamp': current.start_time,
                            'answer': next_seg.text,
                            'answer_speaker': next_seg.speaker,
                            'answer_timestamp': next_seg.start_time,
                            'time_gap': time_gap,
                            'question_index': current.index,
                            'answer_index': next_seg.index
                        })

        logger.info(f"Detected {len(qa_pairs)} question-answer pairs")
        return qa_pairs

    def detect_ire_patterns(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """
        Detect IRE (Initiation-Response-Evaluation) patterns.

        Common in educational discourse: teacher asks, student responds,
        teacher evaluates.

        Args:
            segments: List of transcript segments

        Returns:
            List of IRE pattern instances
        """
        ire_patterns = []

        for i in range(len(segments) - 2):
            initiation = segments[i]
            response = segments[i + 1]
            evaluation = segments[i + 2]

            # Check if it matches IRE pattern
            if (initiation.speaker_role == 'teacher' and
                initiation.is_question and
                response.speaker_role == 'student'):

                # Check if evaluation is from teacher
                eval_result = self.detector.detect_positive_feedback(evaluation.text)
                if (evaluation.speaker_role == 'teacher' and
                    eval_result['is_positive_feedback']):

                    ire_patterns.append({
                        'initiation': initiation.text,
                        'response': response.text,
                        'evaluation': evaluation.text,
                        'timestamp': initiation.start_time,
                        'feedback_type': 'positive' if eval_result['is_positive_feedback'] else 'neutral',
                        'segment_indices': [i, i + 1, i + 2]
                    })

        logger.info(f"Detected {len(ire_patterns)} IRE patterns")
        return ire_patterns

    def detect_peer_interaction(self, segments: List[TranscriptSegment]) -> List[Dict]:
        """
        Detect student-to-student interactions.

        Args:
            segments: List of transcript segments

        Returns:
            List of peer interaction sequences
        """
        peer_interactions = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # Both must be students
            if (current.speaker_role == 'student' and
                next_seg.speaker_role == 'student' and
                current.speaker != next_seg.speaker):

                time_gap = (next_seg.start_time - current.end_time).total_seconds()
                if time_gap < 5:  # Short gap indicates interaction
                    peer_interactions.append({
                        'student1': current.speaker,
                        'student1_text': current.text,
                        'student2': next_seg.speaker,
                        'student2_text': next_seg.text,
                        'timestamp': current.start_time,
                        'segment_indices': [current.index, next_seg.index]
                    })

        logger.info(f"Detected {len(peer_interactions)} peer interactions")
        return peer_interactions
