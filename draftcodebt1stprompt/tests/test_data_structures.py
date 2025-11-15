"""
Test Suite for Data Structures Module
Tests all dataclasses, properties, and methods
"""

import pytest
from datetime import timedelta, datetime
from agile_education_analyzer.data_structures import (
    TranscriptSegment,
    SessionMetadata,
    UkrainianDiscoursePattern,
    AgileTerminology,
    ProblemInstance,
    EngagementMetrics,
    AgileAdoptionMetrics,
    TeachingPattern,
    Insight,
    SessionAnalysisResult,
    SpeakerRole,
    DiscourseType
)

class TestTranscriptSegment:
    """Test TranscriptSegment dataclass"""

    def test_create_basic_segment(self):
        """Test creating a basic transcript segment"""
        segment = TranscriptSegment(
            index=0,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=5),
            text="Test text"
        )
        assert segment.index == 0
        assert segment.text == "Test text"
        assert segment.speaker is None

    def test_segment_duration_property(self):
        """Test duration calculation"""
        segment = TranscriptSegment(
            index=0,
            start_time=timedelta(seconds=10),
            end_time=timedelta(seconds=25),
            text="Test"
        )
        assert segment.duration == timedelta(seconds=15)

    def test_segment_text_length_property(self):
        """Test text length property"""
        segment = TranscriptSegment(
            index=0,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=5),
            text="Hello world"
        )
        assert segment.text_length == 11

    def test_segment_with_all_attributes(self):
        """Test segment with all optional attributes"""
        segment = TranscriptSegment(
            index=5,
            start_time=timedelta(seconds=30),
            end_time=timedelta(seconds=35),
            text="Що таке спринт?",
            speaker="Student_1",
            speaker_role="student",
            sentiment=0.5,
            sentiment_label="positive",
            engagement_score=0.8,
            topics=["agile", "sprint"],
            codes=["BLOOM_UNDERSTAND"],
            agile_terms=["спринт"],
            is_question=True,
            is_confusion=False,
            is_understanding=False,
            technical_terms=["sprint"]
        )
        assert segment.is_question is True
        assert segment.sentiment == 0.5
        assert len(segment.agile_terms) == 1
        assert "agile" in segment.topics

class TestSessionMetadata:
    """Test SessionMetadata dataclass"""

    def test_create_sprint_metadata(self):
        """Test creating sprint session metadata"""
        metadata = SessionMetadata(
            filename="Web2.П02.Спринт 1, частина 1.vtt",
            session_type="sprint",
            sprint_number=1,
            part_number=1
        )
        assert metadata.session_type == "sprint"
        assert metadata.sprint_number == 1

    def test_create_standup_metadata(self):
        """Test creating standup metadata"""
        metadata = SessionMetadata(
            filename="Web2.Стендап 1.vtt",
            session_type="standup",
            sprint_number=1
        )
        assert metadata.session_type == "standup"

    def test_session_id_property_sprint(self):
        """Test session ID generation for sprint"""
        metadata = SessionMetadata(
            filename="test.vtt",
            session_type="sprint",
            sprint_number=2,
            part_number=3
        )
        assert metadata.session_id == "sprint_2_part_3"

    def test_session_id_property_standup(self):
        """Test session ID generation for standup"""
        metadata = SessionMetadata(
            filename="test.vtt",
            session_type="standup",
            sprint_number=1
        )
        assert metadata.session_id == "standup_1"

    def test_session_id_property_introduction(self):
        """Test session ID generation for introduction"""
        metadata = SessionMetadata(
            filename="test.vtt",
            session_type="introduction"
        )
        assert metadata.session_id == "introduction_session"

class TestUkrainianDiscoursePattern:
    """Test UkrainianDiscoursePattern dataclass"""

    def test_question_patterns_exist(self):
        """Test that question patterns are defined"""
        pattern = UkrainianDiscoursePattern()
        assert 'general' in pattern.question_patterns
        assert 'clarification' in pattern.question_patterns
        assert 'technical' in pattern.question_patterns

    def test_confusion_patterns_exist(self):
        """Test that confusion patterns are defined"""
        pattern = UkrainianDiscoursePattern()
        assert len(pattern.confusion_patterns) > 0
        assert any('розумію' in p for p in pattern.confusion_patterns)

    def test_understanding_patterns_exist(self):
        """Test that understanding patterns are defined"""
        pattern = UkrainianDiscoursePattern()
        assert len(pattern.understanding_patterns) > 0
        assert any('зрозуміло' in p for p in pattern.understanding_patterns)

    def test_teacher_patterns_exist(self):
        """Test that teacher patterns are defined"""
        pattern = UkrainianDiscoursePattern()
        assert len(pattern.teacher_explanation_patterns) > 0
        assert len(pattern.teacher_instruction_patterns) > 0

    def test_code_switching_indicators_exist(self):
        """Test that code-switching indicators are defined"""
        pattern = UkrainianDiscoursePattern()
        assert len(pattern.code_switching_indicators) > 0
        assert 'function' in pattern.code_switching_indicators

class TestAgileTerminology:
    """Test AgileTerminology dataclass"""

    def test_basic_terms_exist(self):
        """Test that basic agile terms are defined"""
        terms = AgileTerminology()
        assert 'sprint' in terms.terms
        assert 'standup' in terms.terms
        assert 'backlog' in terms.terms
        assert 'scrum' in terms.terms

    def test_terms_have_ukrainian_variants(self):
        """Test that terms have Ukrainian variants"""
        terms = AgileTerminology()
        assert 'спринт' in terms.terms['sprint']
        assert 'стендап' in terms.terms['standup']

    def test_terms_have_english_variants(self):
        """Test that terms have English variants"""
        terms = AgileTerminology()
        assert 'sprint' in terms.terms['sprint']
        assert 'stand-up' in terms.terms['standup']

    def test_ceremony_terms_exist(self):
        """Test that ceremony terms are defined"""
        terms = AgileTerminology()
        assert 'sprint_planning' in terms.ceremony_terms
        assert 'sprint_review' in terms.ceremony_terms

    def test_role_terms_exist(self):
        """Test that role terms are defined"""
        terms = AgileTerminology()
        assert 'product_owner' in terms.role_terms
        assert 'scrum_master' in terms.role_terms

class TestEnums:
    """Test enumeration classes"""

    def test_speaker_role_enum(self):
        """Test SpeakerRole enum"""
        assert SpeakerRole.TEACHER.value == "teacher"
        assert SpeakerRole.STUDENT.value == "student"
        assert SpeakerRole.UNKNOWN.value == "unknown"

    def test_discourse_type_enum(self):
        """Test DiscourseType enum"""
        assert DiscourseType.QUESTION.value == "question"
        assert DiscourseType.EXPLANATION.value == "explanation"
        assert DiscourseType.CONFUSION.value == "confusion"

class TestProblemInstance:
    """Test ProblemInstance dataclass"""

    def test_create_problem_instance(self):
        """Test creating a problem instance"""
        problem = ProblemInstance(
            category="technical",
            description="Error in function",
            segment_index=10,
            timestamp=timedelta(seconds=120),
            speaker="Student_1",
            severity="high"
        )
        assert problem.category == "technical"
        assert problem.severity == "high"
        assert problem.resolved is None

class TestEngagementMetrics:
    """Test EngagementMetrics dataclass"""

    def test_create_engagement_metrics(self):
        """Test creating engagement metrics"""
        metrics = EngagementMetrics(
            total_segments=100,
            speaker_distribution={'Teacher': 60, 'Student_1': 40},
            question_count=15,
            questions_per_speaker={'Student_1': 15},
            average_segment_length=5.5,
            participation_rate=0.75,
            confusion_instances=3,
            understanding_confirmations=20,
            active_discussion_segments=30
        )
        assert metrics.total_segments == 100
        assert metrics.question_count == 15
        assert metrics.participation_rate == 0.75

class TestAgileAdoptionMetrics:
    """Test AgileAdoptionMetrics dataclass"""

    def test_create_agile_metrics(self):
        """Test creating agile adoption metrics"""
        metrics = AgileAdoptionMetrics(
            total_agile_terms=50,
            unique_terms_used=['sprint', 'standup', 'backlog'],
            term_frequency={'sprint': 20, 'standup': 15, 'backlog': 15},
            student_adoption_rate=0.65,
            teacher_usage_rate=0.85,
            correct_usage_count=45,
            misconception_count=5,
            code_switching_instances=12
        )
        assert metrics.total_agile_terms == 50
        assert len(metrics.unique_terms_used) == 3
        assert metrics.student_adoption_rate == 0.65

class TestTeachingPattern:
    """Test TeachingPattern dataclass"""

    def test_create_teaching_pattern(self):
        """Test creating a teaching pattern"""
        pattern = TeachingPattern(
            pattern_type="explanation",
            segments=[1, 2, 3],
            effectiveness_score=0.8,
            student_response="engaged"
        )
        assert pattern.pattern_type == "explanation"
        assert len(pattern.segments) == 3
        assert pattern.effectiveness_score == 0.8

class TestInsight:
    """Test Insight dataclass"""

    def test_create_insight(self):
        """Test creating an insight"""
        insight = Insight(
            insight_type="breakthrough",
            title="Student understanding increased",
            description="Clear explanation led to understanding",
            evidence=["Student said: зрозуміло"],
            timestamp=timedelta(seconds=300),
            confidence=0.9,
            recommendations=["Continue using visual examples"],
            related_segments=[10, 11, 12]
        )
        assert insight.insight_type == "breakthrough"
        assert insight.confidence == 0.9
        assert len(insight.recommendations) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
