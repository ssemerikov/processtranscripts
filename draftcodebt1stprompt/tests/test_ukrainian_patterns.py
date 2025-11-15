"""
Test Suite for Ukrainian Discourse Pattern Detection
Validates Ukrainian language pattern matching and classification
"""

import pytest
from datetime import timedelta

from agile_education_analyzer.ukrainian_patterns import (
    UkrainianDiscourseDetector,
    DialoguePatternAnalyzer
)
from agile_education_analyzer.data_structures import TranscriptSegment

class TestUkrainianDiscourseDetector:
    """Test Ukrainian discourse pattern detection"""

    @pytest.fixture
    def detector(self):
        """Create detector instance for tests"""
        return UkrainianDiscourseDetector()

    def test_detect_questions_simple(self, detector):
        """Test detection of simple Ukrainian questions"""
        # Simple question with question mark
        result = detector.detect_questions("Що таке спринт?")
        assert result['is_question'] is True
        assert result['confidence'] == 1.0

        # Question without question mark but with pattern
        result = detector.detect_questions("Чи можна використати цей метод")
        assert result['is_question'] is True

    def test_detect_questions_clarification(self, detector):
        """Test detection of clarification questions"""
        text = "Можете пояснити це детальніше?"
        result = detector.detect_questions(text)
        assert result['is_question'] is True
        assert result['question_type'] == 'clarification'

    def test_detect_questions_technical(self, detector):
        """Test detection of technical questions"""
        text = "Як це налаштувати в конфігурації?"
        result = detector.detect_questions(text)
        assert result['is_question'] is True
        assert result['question_type'] == 'technical'

    def test_no_question_detected(self, detector):
        """Test that non-questions are not detected"""
        result = detector.detect_questions("Це простий текст без питання")
        assert result['is_question'] is False

    def test_detect_confusion_simple(self, detector):
        """Test detection of confusion indicators"""
        text = "Не розумію як це працює"
        result = detector.detect_confusion(text)
        assert result['is_confused'] is True
        assert result['indicator_count'] > 0

    def test_detect_confusion_multiple_indicators(self, detector):
        """Test detection with multiple confusion markers"""
        text = "Не розумію, не працює, помилка"
        result = detector.detect_confusion(text)
        assert result['is_confused'] is True
        assert result['confusion_level'] in ['moderate', 'severe']

    def test_no_confusion_detected(self, detector):
        """Test that non-confused text is not detected"""
        result = detector.detect_confusion("Все зрозуміло і працює добре")
        assert result['is_confused'] is False

    def test_detect_understanding_exact_match(self, detector):
        """Test detection of exact understanding confirmations"""
        texts = ["зрозуміло", "так", "добре", "окей"]
        for text in texts:
            result = detector.detect_understanding(text)
            assert result['is_understanding'] is True
            assert result['confidence'] == 1.0

    def test_detect_understanding_pattern(self, detector):
        """Test detection of understanding patterns"""
        text = "Тепер зрозуміло, дякую"
        result = detector.detect_understanding(text)
        assert result['is_understanding'] is True

    def test_detect_teacher_explanation(self, detector):
        """Test detection of teacher explanation patterns"""
        text = "Пояснюю: наприклад, це означає..."
        result = detector.detect_teacher_explanation(text)
        assert result['is_teacher_discourse'] is True
        assert 'explanation' in result['discourse_type']

    def test_detect_teacher_instruction(self, detector):
        """Test detection of teacher instructions"""
        text = "Зробіть завдання до наступного разу"
        result = detector.detect_teacher_explanation(text)
        assert result['is_teacher_discourse'] is True
        assert 'instruction' in result['discourse_type']

    def test_detect_code_switching(self, detector):
        """Test detection of Ukrainian-English code-switching"""
        text = "Використайте function для створення component"
        result = detector.detect_code_switching(text)
        assert result['has_code_switching'] is True
        assert result['english_term_count'] >= 2
        assert 'function' in result['english_terms']
        assert 'component' in result['english_terms']

    def test_no_code_switching(self, detector):
        """Test that pure Ukrainian doesn't trigger code-switching"""
        text = "Використайте функцію для створення компонента"
        result = detector.detect_code_switching(text)
        # May have code-switching due to Latin letters, but low ratio
        assert result['switching_ratio'] < 0.1

    def test_analyze_segment(self, detector):
        """Test comprehensive segment analysis"""
        segment = TranscriptSegment(
            index=0,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=5),
            text="Не розумію, як це працює?",
            speaker="Student_1",
            speaker_role="student"
        )

        updated_segment = detector.analyze_segment(segment)
        assert updated_segment.is_question is True
        assert updated_segment.is_confusion is True
        assert updated_segment.is_understanding is False

    def test_classify_discourse_type(self, detector):
        """Test discourse type classification"""
        # Confused question
        discourse_type = detector.classify_discourse_type(
            "Не розумію, що таке спринт?",
            speaker_role="student"
        )
        assert discourse_type == 'confused_question'

        # Simple confirmation
        discourse_type = detector.classify_discourse_type(
            "Зрозуміло",
            speaker_role="student"
        )
        assert discourse_type == 'confirmation'

        # Teacher explanation
        discourse_type = detector.classify_discourse_type(
            "Пояснюю: спринт це часовий період",
            speaker_role="teacher"
        )
        assert 'explanation' in discourse_type

class TestDialoguePatternAnalyzer:
    """Test dialogue pattern analysis"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return DialoguePatternAnalyzer()

    @pytest.fixture
    def sample_segments(self):
        """Create sample segments for testing"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=3),
                text="Що таке спринт?",
                speaker="Student_1",
                speaker_role="student",
                is_question=True
            ),
            TranscriptSegment(
                index=1,
                start_time=timedelta(seconds=4),
                end_time=timedelta(seconds=8),
                text="Спринт це часовий період для виконання завдань",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=False
            ),
            TranscriptSegment(
                index=2,
                start_time=timedelta(seconds=9),
                end_time=timedelta(seconds=11),
                text="Правильно, молодець",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=False
            )
        ]
        return segments

    def test_detect_qa_pairs(self, analyzer, sample_segments):
        """Test Q&A pair detection"""
        qa_pairs = analyzer.detect_qa_pairs(sample_segments)
        assert len(qa_pairs) >= 1
        assert qa_pairs[0]['question_speaker'] == "Student_1"
        assert qa_pairs[0]['answer_speaker'] == "Teacher"

    def test_detect_ire_patterns(self, analyzer):
        """Test IRE (Initiation-Response-Evaluation) pattern detection"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=2),
                text="Хто знає що таке backlog?",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=True
            ),
            TranscriptSegment(
                index=1,
                start_time=timedelta(seconds=3),
                end_time=timedelta(seconds=5),
                text="Це список завдань для спринту",
                speaker="Student_1",
                speaker_role="student",
                is_question=False
            ),
            TranscriptSegment(
                index=2,
                start_time=timedelta(seconds=6),
                end_time=timedelta(seconds=8),
                text="Правильно, молодець",
                speaker="Teacher",
                speaker_role="teacher",
                is_question=False
            )
        ]

        ire_patterns = analyzer.detect_ire_patterns(segments)
        assert len(ire_patterns) >= 1
        assert ire_patterns[0]['feedback_type'] == 'positive'

    def test_detect_peer_interaction(self, analyzer):
        """Test peer-to-peer interaction detection"""
        segments = [
            TranscriptSegment(
                index=0,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=2),
                text="Я думаю треба використати метод X",
                speaker="Student_1",
                speaker_role="student",
                is_question=False
            ),
            TranscriptSegment(
                index=1,
                start_time=timedelta(seconds=3),
                end_time=timedelta(seconds=5),
                text="Так, давай спробуємо",
                speaker="Student_2",
                speaker_role="student",
                is_question=False
            )
        ]

        peer_interactions = analyzer.detect_peer_interaction(segments)
        assert len(peer_interactions) >= 1
        assert peer_interactions[0]['student1'] == "Student_1"
        assert peer_interactions[0]['student2'] == "Student_2"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
