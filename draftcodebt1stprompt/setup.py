"""
Setup configuration for Agile Education Analysis Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "USAGE_GUIDE.md").read_text(encoding='utf-8')

setup(
    name='agile-education-analyzer',
    version='1.0.0',
    author='Agile Education Research Team',
    author_email='research@example.com',
    description='Framework for analyzing Ukrainian educational transcripts from agile programming courses',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/agile-education-analyzer',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: Ukrainian',
        'Natural Language :: English',
    ],
    python_requires='>=3.8',
    install_requires=[
        # Core Data Processing
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'webvtt-py>=0.4.6',
        'python-dateutil>=2.8.2',

        # Ukrainian NLP Support
        'spacy>=3.4.0',
        'stanza>=1.4.0',
        'pymorphy3>=1.2.0',
        'pymorphy3-dicts-uk>=2.4.1',

        # Transformer Models
        'transformers>=4.30.0',
        'sentence-transformers>=2.2.0',
        'torch>=2.0.0',

        # Text Analysis
        'scikit-learn>=1.3.0',
        'gensim>=4.3.0',
        'nltk>=3.8',

        # Statistical Analysis
        'scipy>=1.10.0',
        'statsmodels>=0.14.0',

        # Visualization
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'plotly>=5.14.0',
        'wordcloud>=1.9.0',
        'networkx>=3.0',

        # Utilities
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'jsonlines>=3.1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'ipython>=8.0.0',
            'jupyter>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'sphinx-autodoc-typehints>=1.22.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'agile-analyzer=agile_education_analyzer.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'agile_education_analyzer': [
            'data/*.json',
            'data/*.yaml',
        ],
    },
    zip_safe=False,
    keywords='education research agile ukrainian nlp transcripts',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/agile-education-analyzer/issues',
        'Source': 'https://github.com/yourusername/agile-education-analyzer',
        'Documentation': 'https://agile-education-analyzer.readthedocs.io',
    },
)
