"""
Visualization Module for Agile Education Analysis Framework
Publication-ready visualizations with Ukrainian text support
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

from .data_structures import SessionAnalysisResult, TranscriptSegment
from .utils.logger import get_logger

logger = get_logger('visualization')

class ResearchVisualizer:
    """
    Create publication-ready visualizations for educational research.

    Handles Ukrainian text rendering, exports in multiple formats,
    and generates academic-quality charts for research papers.
    """

    def __init__(self, style: str = 'seaborn-v0_8-paper', dpi: int = 300,
                 font_family: str = 'DejaVu Sans'):
        """
        Initialize visualizer with publication settings.

        Args:
            style: Matplotlib style
            dpi: Resolution for raster exports
            font_family: Font supporting Cyrillic (Ukrainian) characters
        """
        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not found, using default")
            plt.style.use('default')

        self.dpi = dpi
        self.font_family = font_family

        # Configure matplotlib for Ukrainian text
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False  # Handle minus sign

        # Color palettes
        self.colors = sns.color_palette("husl", 12)
        self.diverging_colors = sns.color_palette("RdYlGn", 11)
        self.sequential_colors = sns.color_palette("viridis", 10)

        logger.info(f"Visualizer initialized: {style}, {dpi} DPI, font={font_family}")

    def plot_engagement_evolution(self, evolution_data: pd.DataFrame,
                                   output_path: str, format: str = 'png'):
        """
        Plot engagement metrics evolution across sprints.

        Args:
            evolution_data: DataFrame with engagement metrics by session
            output_path: Output file path
            format: Output format ('png', 'svg', 'pdf')
        """
        logger.info("Creating engagement evolution plot")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('avg_questions_per_student', 'Питання на студента', 'Questions per Student'),
            ('participation_rate', 'Рівень участі', 'Participation Rate'),
            ('confusion_rate', 'Рівень плутанини', 'Confusion Rate'),
            ('understanding_rate', 'Рівень розуміння', 'Understanding Rate')
        ]

        for ax, (metric, title_uk, title_en) in zip(axes.flat, metrics):
            sprint_data = evolution_data[evolution_data['sprint_number'].notna()]

            if len(sprint_data) > 0:
                ax.plot(sprint_data['sprint_number'], sprint_data[metric],
                       marker='o', linewidth=2.5, markersize=10,
                       color=self.colors[0], alpha=0.8)

                # Add trend line
                z = np.polyfit(sprint_data['sprint_number'], sprint_data[metric], 1)
                p = np.poly1d(z)
                ax.plot(sprint_data['sprint_number'], p(sprint_data['sprint_number']),
                       "--", alpha=0.5, color=self.colors[1], label='Trend')

                ax.set_xlabel('Номер спринту / Sprint Number', fontsize=11)
                ax.set_ylabel(f'{title_uk} / {title_en}', fontsize=11)
                ax.set_title(title_uk, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend()

                # Format y-axis as percentage if it's a rate
                if 'rate' in metric:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()
        self._save_figure(fig, output_path, format)
        logger.info(f"Engagement evolution plot saved to {output_path}")

    def plot_agile_adoption(self, adoption_data: pd.DataFrame,
                           output_path: str, format: str = 'png'):
        """
        Plot agile terminology adoption patterns.

        Args:
            adoption_data: DataFrame with agile metrics
            output_path: Output file path
            format: Output format
        """
        logger.info("Creating agile adoption plot")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sprint_data = adoption_data[adoption_data['sprint_number'].notna()]

        if len(sprint_data) > 0:
            # Adoption rate over time
            ax1.plot(sprint_data['sprint_number'],
                    sprint_data['student_adoption_rate'],
                    marker='s', label='Студенти / Students',
                    linewidth=2.5, markersize=10, color=self.colors[0])
            ax1.plot(sprint_data['sprint_number'],
                    sprint_data['correct_usage_rate'],
                    marker='^', label='Правильне використання / Correct Usage',
                    linewidth=2.5, markersize=10, color=self.colors[2])

            ax1.set_xlabel('Номер спринту / Sprint Number', fontsize=12)
            ax1.set_ylabel('Рівень / Rate', fontsize=12)
            ax1.set_title('Прийняття термінології Agile / Agile Terminology Adoption',
                         fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Term frequency by session type
        if 'session_type' in adoption_data.columns:
            session_types = adoption_data.groupby('session_type')['total_agile_terms'].mean()
            bars = ax2.bar(session_types.index, session_types.values,
                          color=self.colors[:len(session_types)], alpha=0.8, edgecolor='black')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10)

            ax2.set_xlabel('Тип сесії / Session Type', fontsize=12)
            ax2.set_ylabel('Середня кількість термінів / Avg Terms Used', fontsize=12)
            ax2.set_title('Використання термінології за типом сесії',
                         fontsize=13, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, output_path, format)
        logger.info(f"Agile adoption plot saved to {output_path}")

    def create_problem_heatmap(self, problem_patterns: pd.DataFrame,
                               output_path: str, format: str = 'png'):
        """
        Create heatmap of problems across sessions.

        Args:
            problem_patterns: DataFrame with problem counts
            output_path: Output file path
            format: Output format
        """
        logger.info("Creating problem heatmap")

        problem_categories = [
            'technical_issues_count',
            'conceptual_difficulties_count',
            'process_challenges_count',
            'collaboration_issues_count'
        ]

        # Prepare data for heatmap
        if all(col in problem_patterns.columns for col in problem_categories):
            heatmap_data = problem_patterns[['session'] + problem_categories].copy()
            heatmap_data = heatmap_data.set_index('session')[problem_categories].T

            # Rename categories to Ukrainian/English
            category_labels = {
                'technical_issues_count': 'Технічні проблеми\nTechnical Issues',
                'conceptual_difficulties_count': 'Концептуальні труднощі\nConceptual Difficulties',
                'process_challenges_count': 'Процесні виклики\nProcess Challenges',
                'collaboration_issues_count': 'Проблеми співпраці\nCollaboration Issues'
            }
            heatmap_data.index = [category_labels.get(idx, idx) for idx in heatmap_data.index]

            plt.figure(figsize=(14, 7))
            sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd',
                       cbar_kws={'label': 'Кількість проблем / Problem Count'},
                       linewidths=0.5, linecolor='gray')

            plt.title('Розподіл проблем по сесіях / Problem Distribution Across Sessions',
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Сесія / Session', fontsize=12)
            plt.ylabel('Категорія проблеми / Problem Category', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            self._save_figure(plt.gcf(), output_path, format)
            logger.info(f"Problem heatmap saved to {output_path}")
        else:
            logger.warning("Missing required columns for problem heatmap")

    def create_speaker_network(self, interactions: List[Dict],
                               output_path: str, format: str = 'png'):
        """
        Create network visualization of speaker interactions.

        Args:
            interactions: List of interaction dictionaries
            output_path: Output file path
            format: Output format
        """
        logger.info("Creating speaker network visualization")

        G = nx.DiGraph()

        # Count interactions between speakers
        interaction_counts = defaultdict(int)
        for interaction in interactions:
            source = interaction.get('source', 'Unknown')
            target = interaction.get('target', 'Unknown')
            interaction_counts[(source, target)] += 1

        # Add edges with weights
        for (source, target), count in interaction_counts.items():
            G.add_edge(source, target, weight=count)

        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        plt.figure(figsize=(12, 9))

        # Draw nodes with size based on degree
        node_sizes = [G.degree(node) * 500 + 1000 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color=self.colors[:len(G.nodes())],
                              alpha=0.9, edgecolors='black', linewidths=2)

        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [5 * (w / max_weight) for w in weights]

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                              arrows=True, arrowsize=20, arrowstyle='->')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold',
                               font_family=self.font_family)

        # Add edge labels (interaction counts)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)

        plt.title('Мережа взаємодії викладач-студент / Teacher-Student Interaction Network',
                 fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        self._save_figure(plt.gcf(), output_path, format)
        logger.info(f"Speaker network saved to {output_path}")

    def create_wordcloud_by_sprint(self, sessions: List[Dict],
                                   output_path: str, format: str = 'png',
                                   stopwords: Optional[List[str]] = None):
        """
        Create word clouds for each sprint with Ukrainian text support.

        Args:
            sessions: List of session dictionaries
            output_path: Output file path
            format: Output format
            stopwords: Optional list of Ukrainian stopwords
        """
        logger.info("Creating sprint word clouds")

        # Default Ukrainian stopwords
        if stopwords is None:
            stopwords = [
                'і', 'в', 'на', 'з', 'у', 'та', 'що', 'це', 'не', 'як',
                'до', 'за', 'по', 'від', 'для', 'є', 'так', 'але', 'або',
                'ми', 'ви', 'вони', 'він', 'вона', 'який', 'яка', 'яке'
            ]

        # Find unique sprints
        sprint_numbers = sorted(set(
            s['metadata'].sprint_number
            for s in sessions
            if s['metadata'].sprint_number is not None
        ))

        n_sprints = len(sprint_numbers)
        if n_sprints == 0:
            logger.warning("No sprint data found for word clouds")
            return

        fig, axes = plt.subplots(1, n_sprints, figsize=(6*n_sprints, 5))
        if n_sprints == 1:
            axes = [axes]

        for idx, sprint_num in enumerate(sprint_numbers):
            # Collect text from sprint sessions
            sprint_text = ""
            for session in sessions:
                if session['metadata'].sprint_number == sprint_num:
                    for segment in session['segments']:
                        sprint_text += " " + segment.text

            if sprint_text.strip():
                # Create word cloud with Ukrainian font
                try:
                    wordcloud = WordCloud(
                        width=800, height=600,
                        background_color='white',
                        colormap='viridis',
                        stopwords=set(stopwords),
                        font_path=None,  # Use system default that supports Cyrillic
                        max_words=100,
                        relative_scaling=0.5,
                        min_font_size=10
                    ).generate(sprint_text)

                    axes[idx].imshow(wordcloud, interpolation='bilinear')
                    axes[idx].set_title(f'Спринт {sprint_num} / Sprint {sprint_num}',
                                       fontsize=13, fontweight='bold')
                    axes[idx].axis('off')

                except Exception as e:
                    logger.error(f"Error creating wordcloud for sprint {sprint_num}: {e}")
                    axes[idx].text(0.5, 0.5, 'Error generating wordcloud',
                                  ha='center', va='center')
                    axes[idx].axis('off')

        plt.tight_layout()
        self._save_figure(fig, output_path, format)
        logger.info(f"Word clouds saved to {output_path}")

    def plot_sentiment_timeline(self, segments: List[TranscriptSegment],
                                output_path: str, format: str = 'png'):
        """
        Plot sentiment evolution over time within a session.

        Args:
            segments: List of transcript segments
            output_path: Output file path
            format: Output format
        """
        logger.info("Creating sentiment timeline")

        # Extract sentiment data
        times = [(seg.start_time.total_seconds() / 60) for seg in segments if seg.sentiment is not None]
        sentiments = [seg.sentiment for seg in segments if seg.sentiment is not None]

        if not times:
            logger.warning("No sentiment data available")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Scatter plot with color mapping
        scatter = ax.scatter(times, sentiments, c=sentiments,
                           cmap='RdYlGn', s=50, alpha=0.6,
                           edgecolors='black', linewidth=0.5)

        # Add smoothed trend line
        if len(times) > 5:
            z = np.polyfit(times, sentiments, 3)
            p = np.poly1d(z)
            times_smooth = np.linspace(min(times), max(times), 100)
            ax.plot(times_smooth, p(times_smooth), 'r--', alpha=0.5,
                   linewidth=2, label='Тренд / Trend')

        # Add neutral line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        ax.set_xlabel('Час (хвилини) / Time (minutes)', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_title('Динаміка sentiment протягом сесії / Sentiment Timeline',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Настрій / Sentiment', fontsize=10)

        plt.tight_layout()
        self._save_figure(fig, output_path, format)
        logger.info(f"Sentiment timeline saved to {output_path}")

    def plot_comparison_chart(self, data: pd.DataFrame, metric: str,
                             group_by: str, output_path: str,
                             format: str = 'png'):
        """
        Create comparison chart (box plot or violin plot).

        Args:
            data: DataFrame with metrics
            metric: Metric to plot
            group_by: Column to group by
            output_path: Output file path
            format: Output format
        """
        logger.info(f"Creating comparison chart for {metric} by {group_by}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Box plot
        sns.boxplot(data=data, x=group_by, y=metric, ax=ax1,
                   palette=self.colors, showmeans=True)
        ax1.set_title(f'{metric} by {group_by} (Box Plot)',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel(group_by, fontsize=11)
        ax1.set_ylabel(metric, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Violin plot
        sns.violinplot(data=data, x=group_by, y=metric, ax=ax2,
                      palette=self.colors)
        ax2.set_title(f'{metric} by {group_by} (Violin Plot)',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel(group_by, fontsize=11)
        ax2.set_ylabel(metric, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_figure(fig, output_path, format)
        logger.info(f"Comparison chart saved to {output_path}")

    def create_interactive_timeline(self, sessions: List[SessionAnalysisResult],
                                   output_path: str):
        """
        Create interactive timeline using Plotly.

        Args:
            sessions: List of session analysis results
            output_path: Output HTML file path
        """
        logger.info("Creating interactive timeline")

        # Prepare data
        timeline_data = []
        for session in sessions:
            timeline_data.append({
                'Session': session.metadata.session_id,
                'Type': session.metadata.session_type,
                'Sprint': session.metadata.sprint_number or 0,
                'Engagement': session.engagement_metrics.participation_rate,
                'Questions': session.engagement_metrics.question_count,
                'Problems': len(session.problems),
                'Agile Terms': session.agile_metrics.total_agile_terms
            })

        df = pd.DataFrame(timeline_data)

        # Create figure with secondary y-axis
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Session'], y=df['Engagement'],
            name='Participation Rate',
            mode='lines+markers',
            line=dict(width=3)
        ))

        fig.add_trace(go.Bar(
            x=df['Session'], y=df['Questions'],
            name='Questions',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Session Metrics Timeline',
            xaxis=dict(title='Session'),
            yaxis=dict(title='Participation Rate'),
            yaxis2=dict(title='Question Count', overlaying='y', side='right'),
            hovermode='x unified',
            height=600
        )

        fig.write_html(output_path)
        logger.info(f"Interactive timeline saved to {output_path}")

    def _save_figure(self, fig, output_path: str, format: str = 'png'):
        """
        Save figure in specified format(s).

        Args:
            fig: Matplotlib figure
            output_path: Base output path
            format: Format (png, svg, pdf, or 'all')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        formats = ['png', 'svg', 'pdf'] if format == 'all' else [format]

        for fmt in formats:
            save_path = output_path.with_suffix(f'.{fmt}')
            try:
                if fmt == 'png':
                    fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                else:
                    fig.savefig(save_path, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                logger.debug(f"Saved figure as {fmt}: {save_path}")
            except Exception as e:
                logger.error(f"Error saving figure as {fmt}: {e}")

        plt.close(fig)
