"""
Pytest configuration and shared fixtures
"""

import pytest
from pathlib import Path
from datetime import timedelta

@pytest.fixture
def sample_transcript_path():
    """Path to sample VTT file for testing"""
    return Path(__file__).parent / 'data' / 'sample.vtt'

@pytest.fixture
def sample_text_ukrainian():
    """Sample Ukrainian text for testing"""
    return "Сьогодні ми почнемо вивчати Scrum методологію. Що таке спринт?"

@pytest.fixture
def sample_segments_list():
    """Sample list of transcript segments"""
    from agile_education_analyzer.data_structures import TranscriptSegment

    return [
        TranscriptSegment(
            index=0,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=5),
            text="Доброго дня, студенти",
            speaker="Teacher",
            speaker_role="teacher"
        ),
        TranscriptSegment(
            index=1,
            start_time=timedelta(seconds=6),
            end_time=timedelta(seconds=10),
            text="Доброго дня",
            speaker="Student_1",
            speaker_role="student"
        ),
        TranscriptSegment(
            index=2,
            start_time=timedelta(seconds=11),
            end_time=timedelta(seconds=15),
            text="Що таке агіль?",
            speaker="Student_2",
            speaker_role="student",
            is_question=True
        )
    ]

@pytest.fixture
def mock_session_metadata():
    """Sample session metadata"""
    from agile_education_analyzer.data_structures import SessionMetadata

    return SessionMetadata(
        filename="Web2.П02.Спринт 1, частина 1.vtt",
        session_type="sprint",
        sprint_number=1,
        part_number=1,
        total_segments=100,
        teacher_segments=60,
        student_segments=40
    )
