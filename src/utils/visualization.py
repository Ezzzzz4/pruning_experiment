"""
Visualization Utilities

Publication-quality figures for layer importance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns


class PruningVisualizer:
    """
    Generate publication-quality visualizations for pruning analysis.
    
    Example:
        >>> viz = PruningVisualizer()
        >>> fig = viz.layer_importance_heatmap(bi_scores, 'GPT-2')
        >>> fig.savefig('figures/gpt2_importance.png', dpi=300)
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer with matplotlib style.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-whitegrid')
        
        # Color scheme for importance levels
        self.colors = {
            'critical': '#e74c3c',      # Red
            'important': '#f39c12',     # Orange
            'moderate': '#f1c40f',      # Yellow
            'redundant': '#2ecc71',     # Green
        }
        
    def _get_importance_color(self, bi_score: float) -> str:
        """Map BI score to importance color."""
        if bi_score >= 0.4:
            return self.colors['critical']
        elif bi_score >= 0.2:
            return self.colors['important']
        elif bi_score >= 0.1:
            return self.colors['moderate']
        else:
            return self.colors['redundant']
    
    def layer_importance_heatmap(
        self,
        bi_scores: Dict[int, float],
        model_name: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Create horizontal bar chart showing layer importance.
        
        Args:
            bi_scores: Dict mapping layer_idx -> BI score
            model_name: Name of model for title
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        layers = sorted(bi_scores.keys())
        scores = [bi_scores[i] for i in layers]
        colors = [self._get_importance_color(s) for s in scores]
        
        y_pos = np.arange(len(layers))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Layer {i}' for i in layers])
        ax.invert_yaxis()  # Top to bottom
        ax.set_xlabel('Block Influence Score')
        ax.set_title(f'Layer Importance: {model_name}')
        ax.set_xlim(0, max(scores) * 1.1)
        
        # Add value annotations
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            label = f'{score:.3f}'
            ax.annotate(
                label,
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords='offset points',
                va='center',
                fontsize=8
            )
        
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['critical'], label='Critical (BI≥0.4)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['important'], label='Important (0.2≤BI<0.4)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['moderate'], label='Moderate (0.1≤BI<0.2)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['redundant'], label='Redundant (BI<0.1)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def cross_architecture_comparison(
        self,
        all_results: Dict[str, Dict[int, float]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Create side-by-side comparison of multiple architectures.
        
        Args:
            all_results: Dict of {model_name: {layer_idx: bi_score}}
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, bi_scores) in zip(axes, all_results.items()):
            layers = sorted(bi_scores.keys())
            scores = [bi_scores[i] for i in layers]
            colors = [self._get_importance_color(s) for s in scores]
            
            y_pos = np.arange(len(layers))
            ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'{i}' for i in layers])
            ax.invert_yaxis()
            ax.set_xlabel('BI Score')
            ax.set_title(model_name)
            ax.set_xlim(0, 1)
        
        plt.suptitle('Cross-Architecture Layer Importance Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def bathtub_curve(
        self,
        bi_scores: Dict[int, float],
        model_name: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create "bathtub curve" showing layer position vs importance.
        
        This visualization demonstrates the expected pattern:
        - High importance at beginning (layers 0-2)
        - Low importance in middle (redundant)
        - High importance at end (output formation)
        
        Args:
            bi_scores: Dict mapping layer_idx -> BI score
            model_name: Name of model
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        layers = sorted(bi_scores.keys())
        scores = [bi_scores[i] for i in layers]
        n_layers = len(layers)
        
        # Normalize layer positions to 0-100%
        positions = [(i / (n_layers - 1)) * 100 for i in range(n_layers)]
        
        # Plot
        ax.plot(positions, scores, 'b-', linewidth=2, marker='o', markersize=8)
        ax.fill_between(positions, scores, alpha=0.3)
        
        # Add horizontal line for "redundant" threshold
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Redundancy threshold')
        
        ax.set_xlabel('Layer Position (%)')
        ax.set_ylabel('Block Influence Score')
        ax.set_title(f'Layer Importance Distribution: {model_name}')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(scores) * 1.1)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def pareto_frontier(
        self,
        results: List[Tuple[float, float, str]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create Pareto frontier plot for speed vs quality trade-off.
        
        Args:
            results: List of (speedup_%, quality_loss_%, label) tuples
            save_path: Optional save path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        speedups = [r[0] for r in results]
        quality_losses = [r[1] for r in results]
        labels = [r[2] for r in results]
        
        # Scatter plot
        scatter = ax.scatter(speedups, quality_losses, s=100, c='blue', alpha=0.7)
        
        # Add labels
        for x, y, label in zip(speedups, quality_losses, labels):
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Compute and plot Pareto frontier
        pareto_points = self._compute_pareto_frontier(list(zip(speedups, quality_losses)))
        if len(pareto_points) > 1:
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto frontier')
        
        ax.set_xlabel('Speedup (%)')
        ax.set_ylabel('Quality Loss (%)')
        ax.set_title('Speed vs Quality Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _compute_pareto_frontier(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Compute Pareto frontier (maximize speedup, minimize quality loss)."""
        # Sort by speedup descending
        sorted_points = sorted(points, key=lambda x: x[0], reverse=True)
        
        pareto = []
        min_loss = float('inf')
        
        for speedup, loss in sorted_points:
            if loss < min_loss:
                pareto.append((speedup, loss))
                min_loss = loss
        
        return sorted(pareto, key=lambda x: x[0])
