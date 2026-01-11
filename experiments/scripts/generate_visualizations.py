"""
Visualization Generator for Language Model Experiments

Generates publication-quality figures:
- Layer importance heatmaps
- Bathtub curves
- Comparison plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150


def load_results(results_dir: Path) -> Dict:
    """Load all result files from a directory."""
    results = {}
    for file in results_dir.glob("*_results.json"):
        with open(file) as f:
            data = json.load(f)
            model_name = data['metadata']['model_name']
            results[model_name] = data
    return results


def plot_layer_heatmap(
    model_name: str,
    bi_scores: Dict[str, float],
    output_path: Path,
    component: str = 'main'
) -> None:
    """
    Create horizontal bar heatmap of layer importance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to numpy array
    layers = sorted([int(k) for k in bi_scores.keys()])
    scores = [bi_scores[str(l)] for l in layers]
    
    # Color mapping
    colors = []
    for score in scores:
        if score < 0.1:
            colors.append('#2E8B57')  # Green - redundant
        elif score < 0.3:
            colors.append('#FFD700')  # Yellow - moderate
        else:
            colors.append('#DC143C')  # Red - important
    
    # Create horizontal bars
    bars = ax.barh(layers, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (layer, score) in enumerate(zip(layers, scores)):
        ax.text(score + 0.02, layer, f'{score:.3f}', va='center', fontsize=10)
    
    # Styling
    ax.set_xlabel('Block Influence (BI)')
    ax.set_ylabel('Layer Index')
    ax.set_title(f'{model_name} - Layer Importance Heatmap ({component})')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.set_yticks(layers)
    ax.invert_yaxis()  # Layer 0 at top
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#DC143C', edgecolor='black', label='Important (BI ≥ 0.3)'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Moderate (0.1 ≤ BI < 0.3)'),
        mpatches.Patch(facecolor='#2E8B57', edgecolor='black', label='Redundant (BI < 0.1)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_bathtub_curve(
    model_name: str,
    bi_scores: Dict[str, float],
    output_path: Path,
    component: str = 'main'
) -> None:
    """
    Create bathtub curve (normalized position vs BI).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy array
    layers = sorted([int(k) for k in bi_scores.keys()])
    scores = [bi_scores[str(l)] for l in layers]
    
    # Normalize positions to [0, 1]
    n_layers = len(layers)
    positions = [l / (n_layers - 1) for l in layers]
    
    # Plot
    ax.plot(positions, scores, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax.fill_between(positions, scores, alpha=0.2, color='#2196F3')
    
    # Add threshold line
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5, label='Redundancy threshold (0.1)')
    
    # Styling
    ax.set_xlabel('Normalized Layer Position (0=first, 1=last)')
    ax.set_ylabel('Block Influence (BI)')
    ax.set_title(f'{model_name} - Bathtub Curve ({component})')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, max(scores) * 1.1)
    ax.legend(loc='upper right')
    
    # Add layer labels
    for i, (pos, score) in enumerate(zip(positions, scores)):
        if score > 0.15 or i == 0 or i == n_layers - 1:
            ax.annotate(f'L{i}', (pos, score), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_summary_card(
    model_name: str,
    results: Dict,
    output_path: Path
) -> None:
    """
    Create an infographic-style summary card.
    """
    stats = results['main']['stats']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, f'{model_name} Analysis Summary', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Stats boxes
    box_props = dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.5)
    
    # Total layers
    ax.text(0.2, 0.7, f"Total Layers\n{stats['total_layers']}", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=box_props)
    
    # Redundant
    ax.text(0.5, 0.7, f"Redundant\n{stats['redundant_count']} ({stats['redundant_pct']:.1f}%)", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.5))
    
    # Mean BI
    ax.text(0.8, 0.7, f"Mean BI\n{stats['mean_bi']:.4f}", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=box_props)
    
    # Layer 2 analysis
    ax.text(0.5, 0.4, f"Layer 2 BI: {stats['layer_2_bi']:.4f}", 
            fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    layer2_status = "✅ CONFIRMED" if stats['layer_2_bi'] > 0.2 else "❌ NOT OBSERVED"
    ax.text(0.5, 0.3, f"Layer 2 Phenomenon: {layer2_status}", 
            fontsize=14, ha='center', va='center', transform=ax.transAxes,
            fontweight='bold')
    
    # Key insight
    ax.text(0.5, 0.15, f"🔍 Key Insight: Layers 3-{stats['total_layers']-2} can likely be pruned",
            fontsize=12, ha='center', va='center', transform=ax.transAxes,
            style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def generate_all_visualizations(
    results_dir: Path = Path("results/data/language"),
    output_dir: Path = Path("results/figures/language")
) -> None:
    """Generate all visualizations for all models."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    print(f"Found {len(results)} model(s)")
    
    for model_name, model_results in results.items():
        print(f"\nGenerating visualizations for {model_name}...")
        
        model_slug = model_name.lower().replace('-', '_').replace('.', '_')
        
        for comp_name in model_results.keys():
            if comp_name == 'metadata':
                continue
            
            comp_data = model_results[comp_name]
            
            # Heatmap
            plot_layer_heatmap(
                model_name=model_name,
                bi_scores=comp_data['bi_scores'],
                output_path=output_dir / f"{model_slug}_{comp_name}_heatmap.png",
                component=comp_name
            )
            
            # Bathtub curve
            plot_bathtub_curve(
                model_name=model_name,
                bi_scores=comp_data['bi_scores'],
                output_path=output_dir / f"{model_slug}_{comp_name}_bathtub.png",
                component=comp_name
            )
        
        # Summary card
        plot_summary_card(
            model_name=model_name,
            results=model_results,
            output_path=output_dir / f"{model_slug}_summary.png"
        )
    
    print("\n✅ All visualizations generated!")


if __name__ == "__main__":
    generate_all_visualizations()
