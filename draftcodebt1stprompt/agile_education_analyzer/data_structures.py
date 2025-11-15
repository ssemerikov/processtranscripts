"""
Data Structures for Agile Education Analysis Framework
Core data models for transcript analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

@dataclass
class TranscriptSegment:
    """
    Represents a single segment from VTT file.

    This is the fundamental unit of analysis, representing one caption
    from a video transcript with its associated metadata and analysis results.
    """
    index: int
    start_time: timedelta
    end_time: timedelta
    text: str
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None  # 'teacher', 'student'
    sentiment: Optional[float] = None
    sentiment_label: Optional[str] = None  # 'positive', 'negative', 'neutral'
    engagement_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)  # Qualitative codes
    agile_terms: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    is_question: bool = False
    is_confusion: bool = False
    is_understanding: bool = False
    technical_terms: List[str] = field(default_factory=list)

    @property
    def duration(self) -> timedelta:
        """Calculate duration of segment"""
        return self.end_time - self.start_time

    @property
    def text_length(self) -> int:
        """Get text length"""
        return len(self.text)

@dataclass
class SessionMetadata:
    """
    Metadata for each educational session.

    Contains structural information about the session extracted
    from filename and analysis.
    """
    filename: str
    session_type: str  # 'introduction', 'sprint', 'standup'
    sprint_number: Optional[int] = None
    part_number: Optional[int] = None  # For sprint parts (1-3)
    duration: Optional[timedelta] = None
    participant_count: Optional[int] = None
    date: Optional[datetime] = None
    total_segments: int = 0
    teacher_segments: int = 0
    student_segments: int = 0

    @property
    def session_id(self) -> str:
        """Generate unique session identifier"""
        if self.session_type == 'standup':
            return f"standup_{self.sprint_number}"
        elif self.session_type == 'sprint':
            return f"sprint_{self.sprint_number}_part_{self.part_number}"
        return f"{self.session_type}_session"

class SpeakerRole(Enum):
    """Enumeration of speaker roles in educational context"""
    TEACHER = "teacher"
    STUDENT = "student"
    UNKNOWN = "unknown"

class DiscourseType(Enum):
    """Types of educational discourse"""
    EXPLANATION = "explanation"
    QUESTION = "question"
    ANSWER = "answer"
    DISCUSSION = "discussion"
    INSTRUCTION = "instruction"
    FEEDBACK = "feedback"
    CONFUSION = "confusion"
    CONFIRMATION = "confirmation"

@dataclass
class UkrainianDiscoursePattern:
    """
    Ukrainian-specific discourse patterns for educational analysis.

    Contains patterns for identifying different types of educational
    discourse in Ukrainian language transcripts.
    """

    # Question patterns (Ukrainian question phrases)
    question_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'general': [
            r'чи\s+можна', r'чи\s+можу', r'чи\s+потрібно',
            r'що\s+таке', r'що\s+це', r'що\s+означає',
            r'як\s+зробити', r'як\s+використати', r'як\s+це\s+працює',
            r'чому\s+не', r'чому\s+так', r'чому\s+це',
            r'де\s+знайти', r'де\s+це', r'де\s+можна',
            r'коли\s+треба', r'коли\s+використовувати',
            r'хто\s+знає', r'хто\s+може',
            r'навіщо\s+потрібно', r'навіщо\s+це',
            r'\?$'  # Ends with question mark
        ],
        'clarification': [
            r'можете\s+пояснити', r'поясніть\s+будь\s+ласка',
            r'не\s+зовсім\s+зрозуміло', r'не\s+розумію',
            r'уточніть\s+будь\s+ласка', r'можна\s+детальніше'
        ],
        'technical': [
            r'як\s+це\s+налаштувати', r'як\s+це\s+встановити',
            r'яку\s+версію', r'який\s+метод', r'яка\s+функція',
            r'в\s+якому\s+файлі', r'в\s+якій\s+директорії'
        ]
    })

    # Confusion indicators
    confusion_patterns: List[str] = field(default_factory=lambda: [
        r'не\s+розумію', r'не\s+зрозумів', r'не\s+зрозуміла',
        r'не\s+працює', r'не\s+виходить', r'не\s+вийшло',
        r'помилка', r'error', r'проблема',
        r'не\s+знаю', r'не\s+впевнений', r'не\s+впевнена',
        r'складно', r'важко', r'не\s+виходить',
        r'щось\s+не\s+так', r'десь\s+помилка',
        r'чому\s+так', r'що\s+не\s+так',
        r'не\s+те', r'не\s+так',
        r'crash', r'падає', r'зависає'
    ])

    # Understanding confirmations
    understanding_patterns: List[str] = field(default_factory=lambda: [
        r'^зрозуміло$', r'^ясно$', r'^окей$', r'^ok$',
        r'^так$', r'^так\s+так', r'^добре$',
        r'тепер\s+зрозуміло', r'тепер\s+ясно',
        r'дякую', r'спасибі', r'thanks',
        r'розумію', r'зрозумів', r'зрозуміла',
        r'все\s+ясно', r'все\s+зрозуміло',
        r'точно', r'правильно', r'вірно',
        r'а\s+так', r'о\s+так', r'ага'
    ])

    # Teacher explanation indicators
    teacher_explanation_patterns: List[str] = field(default_factory=lambda: [
        r'пояснюю', r'розглядаємо', r'розглянемо',
        r'подивіться', r'дивіться', r'звернімо\s+увагу',
        r'важливо\s+зауважити', r'варто\s+зазначити',
        r'наприклад', r'припустимо', r'скажімо',
        r'тобто', r'іншими\s+словами', r'це\s+означає',
        r'давайте\s+розглянемо', r'розберемо',
        r'як\s+ви\s+бачите', r'як\s+видно',
        r'по-перше', r'по-друге', r'по-третє',
        r'отже', r'таким\s+чином', r'в\s+результаті'
    ])

    # Teacher instruction patterns
    teacher_instruction_patterns: List[str] = field(default_factory=lambda: [
        r'зробіть', r'виконайте', r'напишіть',
        r'створіть', r'додайте', r'видаліть',
        r'потрібно\s+зробити', r'необхідно\s+виконати',
        r'ваше\s+завдання', r'домашнє\s+завдання',
        r'до\s+наступного\s+разу', r'до\s+наступної\s+пари',
        r'працюйте\s+над', r'продовжуйте',
        r'спробуйте', r'подумайте', r'проаналізуйте'
    ])

    # Positive feedback patterns
    positive_feedback_patterns: List[str] = field(default_factory=lambda: [
        r'правильно', r'вірно', r'точно', r'чудово',
        r'молодець', r'молодці', r'добре', r'відмінно',
        r'так\s+так', r'саме\s+так', r'абсолютно\s+вірно',
        r'гарна\s+робота', r'хороша\s+робота',
        r'прогрес', r'покращення', r'краще'
    ])

    # Code-switching indicators (Ukrainian-English technical terms)
    code_switching_indicators: List[str] = field(default_factory=lambda: [
        r'function', r'method', r'class', r'variable',
        r'commit', r'push', r'pull', r'merge',
        r'branch', r'repository', r'git',
        r'array', r'object', r'string', r'number',
        r'loop', r'condition', r'if', r'else',
        r'return', r'import', r'export',
        r'component', r'props', r'state',
        r'hook', r'effect', r'render'
    ])

@dataclass
class AgileTerminology:
    """
    Comprehensive agile terminology for Ukrainian/English code-switching.

    Maps agile concepts to their Ukrainian and English variations
    commonly found in educational transcripts.
    """
    terms: Dict[str, List[str]] = field(default_factory=lambda: {
        'sprint': ['спринт', 'спрінт', 'sprint'],
        'standup': ['стендап', 'станд-ап', 'stand-up', 'щоденна зустріч', 'daily'],
        'backlog': ['беклог', 'бек-лог', 'backlog', 'список завдань', 'бэклог'],
        'scrum': ['скрам', 'scrum', 'скрум'],
        'agile': ['гнучка розробка', 'гнучкий', 'agile', 'аджайл', 'аґайл'],
        'user_story': ['користувацька історія', 'user story', 'юзер сторі', 'історія користувача'],
        'retrospective': ['ретроспектива', 'ретро', 'retrospective', 'ретроспективна зустріч'],
        'velocity': ['швидкість', 'velocity', 'продуктивність команди', 'велосіті'],
        'burndown': ['burndown', 'графік згоряння', 'берндаун', 'burndown chart'],
        'iteration': ['ітерація', 'iteration', 'цикл'],
        'kanban': ['канбан', 'kanban'],
        'product_owner': ['власник продукту', 'product owner', 'PO', 'продакт овнер'],
        'scrum_master': ['скрам майстер', 'scrum master', 'SM', 'скрам-майстер'],
        'team': ['команда', 'team', 'група'],
        'planning': ['планування', 'planning', 'sprint planning'],
        'review': ['огляд', 'review', 'перегляд', 'sprint review'],
        'demo': ['демо', 'демонстрація', 'demo', 'presentation'],
        'epic': ['епік', 'epic', 'велике завдання'],
        'task': ['завдання', 'task', 'задача', 'таск'],
        'bug': ['баг', 'помилка', 'bug', 'дефект', 'issue'],
        'feature': ['функція', 'feature', 'функціональність', 'фіча'],
        'commit': ['коміт', 'commit', 'фіксація', 'комміт'],
        'merge': ['мердж', 'merge', 'злиття', 'об\'єднання'],
        'branch': ['гілка', 'branch', 'бранч', 'вітка'],
        'pull_request': ['пул реквест', 'pull request', 'PR', 'запит на злиття'],
        'daily_standup': ['щоденний стендап', 'daily standup', 'daily meeting'],
        'definition_of_done': ['критерії готовності', 'definition of done', 'DoD'],
        'acceptance_criteria': ['критерії прийняття', 'acceptance criteria'],
        'story_points': ['стор поїнти', 'story points', 'бали за історію'],
        'increment': ['інкремент', 'increment', 'приріст'],
        'timebox': ['таймбокс', 'timebox', 'часові рамки'],
        'blocker': ['блокер', 'blocker', 'перешкода', 'проблема'],
        'stakeholder': ['зацікавлена сторона', 'stakeholder', 'стейкхолдер']
    })

    # Agile ceremony/event specific terminology
    ceremony_terms: Dict[str, List[str]] = field(default_factory=lambda: {
        'sprint_planning': ['планування спринту', 'sprint planning'],
        'sprint_review': ['огляд спринту', 'sprint review'],
        'sprint_retrospective': ['ретроспектива спринту', 'sprint retrospective', 'ретро'],
        'daily_standup': ['щоденний стендап', 'daily standup', 'daily scrum'],
        'backlog_refinement': ['уточнення беклогу', 'backlog refinement', 'grooming']
    })

    # Role-specific terminology
    role_terms: Dict[str, List[str]] = field(default_factory=lambda: {
        'product_owner': ['власник продукту', 'PO', 'product owner'],
        'scrum_master': ['скрам-майстер', 'SM', 'scrum master'],
        'development_team': ['команда розробки', 'development team', 'dev team'],
        'stakeholder': ['зацікавлена сторона', 'stakeholder']
    })

@dataclass
class ProblemInstance:
    """Represents an identified problem or challenge"""
    category: str  # 'technical', 'conceptual', 'process', 'collaboration', 'tools'
    description: str
    segment_index: int
    timestamp: timedelta
    speaker: Optional[str] = None
    severity: str = 'medium'  # 'low', 'medium', 'high'
    resolved: Optional[bool] = None
    resolution_method: Optional[str] = None  # 'teacher_help', 'peer_support', 'self_solved'

@dataclass
class EngagementMetrics:
    """Engagement metrics for a session"""
    total_segments: int
    speaker_distribution: Dict[str, int]
    question_count: int
    questions_per_speaker: Dict[str, int]
    average_segment_length: float
    participation_rate: float
    confusion_instances: int
    understanding_confirmations: int
    active_discussion_segments: int

@dataclass
class AgileAdoptionMetrics:
    """Metrics for agile terminology adoption"""
    total_agile_terms: int
    unique_terms_used: List[str]
    term_frequency: Dict[str, int]
    student_adoption_rate: float
    teacher_usage_rate: float
    correct_usage_count: int
    misconception_count: int
    code_switching_instances: int

@dataclass
class TeachingPattern:
    """Identified teaching pattern"""
    pattern_type: str  # 'explanation', 'demonstration', 'questioning', 'feedback'
    segments: List[int]  # Segment indices
    effectiveness_score: Optional[float] = None
    student_response: Optional[str] = None  # 'engaged', 'confused', 'neutral'

@dataclass
class Insight:
    """Automated insight generated from analysis"""
    insight_type: str  # 'breakthrough', 'confusion_point', 'teaching_opportunity'
    title: str
    description: str
    evidence: List[str]  # Segment texts or quotes
    timestamp: timedelta
    confidence: float  # 0.0 to 1.0
    recommendations: List[str] = field(default_factory=list)
    related_segments: List[int] = field(default_factory=list)

@dataclass
class SessionAnalysisResult:
    """Complete analysis results for a session"""
    metadata: SessionMetadata
    segments: List[TranscriptSegment]
    engagement_metrics: EngagementMetrics
    agile_metrics: AgileAdoptionMetrics
    problems: List[ProblemInstance]
    teaching_patterns: List[TeachingPattern]
    insights: List[Insight]
    sentiment_summary: Dict[str, float]
    topics: List[Dict[str, Any]]
    statistical_summary: Dict[str, Any]
