# Configuration for Agile Education Analysis Framework
# Research parameters and settings

import yaml
from pathlib import Path

# ============================================================================
# RESEARCH CONFIGURATION
# ============================================================================

config = {
    # Data paths
    'paths': {
        'transcripts_dir': '/mnt/project',
        'output_dir': '/mnt/user-data/outputs',
        'visualizations_dir': '/mnt/user-data/outputs/visualizations',
        'reports_dir': '/mnt/user-data/outputs/reports',
        'cache_dir': '/mnt/user-data/outputs/cache'
    },
    
    # Ukrainian language settings
    'language': {
        'code': 'uk',
        'encoding': 'utf-8',
        'spacy_model': 'uk_core_news_sm',  # Install with: python -m spacy download uk_core_news_sm
        'stanza_model': 'uk',
        'sentiment_model': 'youscan/ukr-roberta-sentiment'
    },
    
    # Session types and structure
    'session_types': {
        'introduction': {
            'pattern': r'Вступ',
            'expected_duration_min': 60,
            'expected_participants': 15
        },
        'sprint': {
            'pattern': r'Спринт',
            'parts_per_sprint': 3,
            'expected_duration_min': 90,
            'expected_participants': 15
        },
        'standup': {
            'pattern': r'Стендап',
            'expected_duration_min': 15,
            'expected_participants': 10,
            'standups_per_sprint': 3
        }
    },
    
    # Analysis parameters
    'analysis': {
        'speaker_diarization': {
            'clustering_eps': 0.5,
            'min_samples': 2,
            'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2'
        },
        'topic_modeling': {
            'n_topics': 10,
            'max_features': 100,
            'ngram_range': (1, 2),
            'coherence_measure': 'c_v'
        },
        'sentiment_analysis': {
            'max_text_length': 512,
            'batch_size': 32
        },
        'problem_detection': {
            'similarity_threshold': 0.7,
            'min_recurring_count': 2
        },
        'engagement': {
            'min_question_length': 10,
            'participation_threshold': 0.3
        }
    },
    
    # Statistical testing
    'statistics': {
        'significance_level': 0.05,
        'use_bonferroni': True,
        'bootstrap_iterations': 1000,
        'effect_size_type': 'cohen_d'
    },
    
    # Visualization settings
    'visualization': {
        'style': 'seaborn',
        'figure_dpi': 300,
        'color_palette': 'husl',
        'font_size': 12,
        'save_format': 'png'
    },
    
    # Research questions and hypotheses
    'research_questions': [
        {
            'id': 'RQ1',
            'question': 'How does student participation evolve across sprints?',
            'metrics': ['participation_rate', 'question_frequency', 'utterance_count'],
            'hypothesis': 'Student participation increases as they become familiar with agile methods'
        },
        {
            'id': 'RQ2', 
            'question': 'What agile concepts are most/least understood?',
            'metrics': ['correct_usage_rate', 'misconception_rate', 'term_frequency'],
            'hypothesis': 'Basic agile terms are well understood, but advanced concepts show misconceptions'
        },
        {
            'id': 'RQ3',
            'question': 'What technical challenges emerge repeatedly?',
            'metrics': ['problem_frequency', 'problem_categories', 'resolution_time'],
            'hypothesis': 'Technical setup issues are most common in early sprints'
        },
        {
            'id': 'RQ4',
            'question': 'How effective are stand-ups for student learning?',
            'metrics': ['standup_engagement', 'problem_reporting', 'peer_interaction'],
            'hypothesis': 'Stand-ups improve problem visibility and peer learning'
        }
    ],
    
    # Qualitative coding schemes
    'coding_schemes': {
        'engagement_levels': {
            'high': ['active questioning', 'solution proposals', 'peer helping'],
            'medium': ['answering questions', 'following instructions', 'basic participation'],
            'low': ['passive listening', 'minimal responses', 'no questions']
        },
        'learning_indicators': {
            'comprehension': ['explains concept', 'applies knowledge', 'makes connections'],
            'confusion': ['asks for clarification', 'expresses difficulty', 'misunderstands'],
            'progress': ['completes task', 'shows improvement', 'overcomes challenge']
        },
        'collaboration_patterns': {
            'cooperative': ['shares knowledge', 'helps peers', 'team discussion'],
            'individual': ['works alone', 'independent solution', 'personal struggle'],
            'conflict': ['disagreement', 'different approaches', 'team tension']
        }
    },
    
    # Export settings
    'export': {
        'formats': ['csv', 'json', 'excel', 'markdown'],
        'include_raw_data': False,
        'anonymize_speakers': True,
        'timestamp_format': '%H:%M:%S'
    }
}

def save_config(filepath: str = '/mnt/user-data/outputs/config.yaml'):
    """Save configuration to YAML file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str = '/mnt/user-data/outputs/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Custom configuration for specific research needs
class ResearchConfig:
    """Research-specific configuration manager"""
    
    def __init__(self, config_path: str = None):
        if config_path and Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            self.config = config
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set nested configuration value"""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def validate(self) -> bool:
        """Validate configuration completeness"""
        required_keys = [
            'paths.transcripts_dir',
            'language.code',
            'analysis.topic_modeling.n_topics',
            'statistics.significance_level'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required configuration: {key}")
                return False
        
        return True

# Initialize default configuration
if __name__ == "__main__":
    save_config()
    
    # Example usage
    research_config = ResearchConfig()
    print(f"Transcripts directory: {research_config.get('paths.transcripts_dir')}")
    print(f"Number of topics: {research_config.get('analysis.topic_modeling.n_topics')}")
    print(f"Configuration valid: {research_config.validate()}")
