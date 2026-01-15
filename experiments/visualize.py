"""
Unified Visualization Generator

Generates publication-quality figures for all model types:
- Layer importance heatmaps
- Bathtub curves (normalized BI profiles)
- Cross-model comparisons
- Summary cards

Usage:
    python visualize.py --type language    # Language model figures
    python visualize.py --type reasoning   # Reasoning model figures  
    python visualize.py --type comparison  # Cross-type comparisons
    python visualize.py --type all         # Generate all figures
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 150


# ========== LOADERS ==========

def load_results(results_dir: Path) -> Dict:
    """Load all result files from a directory."""
    results = {}
    if not results_dir.exists():
        return results
        
    # Support both naming conventions
    files = list(results_dir.glob("*_benchmark.json")) + list(results_dir.glob("*_results.json"))
    
    for file in files:
        with open(file) as f:
            try:
                data = json.load(f)
                # Handle different JSON structures
                if 'metadata' in data:
                    model_name = data['metadata']['model_name']
                elif 'model_name' in data:
                    model_name = data['model_name']
                else:
                    model_name = file.stem.replace('_results', '').replace('_benchmark', '')
                
                # Normalize keys if needed
                if 'configurations' in data and 'results' not in data:
                    data['results'] = data.pop('configurations')
                    
                if 'results' in data:
                    for res in data['results']:
                        # Fix layer count key
                        # Priority 1: Use 'n_removed' if available (explicit count)
                        if 'n_removed' in res:
                             res['layers_removed'] = res['n_removed']
                        # Priority 2: Use 'layer_idx' if 'layers_removed' missing
                        elif 'layers_removed' not in res and 'layer_idx' in res:
                            res['layers_removed'] = res['layer_idx']
                        
                        # Fix: if layers_removed is a list, use its length or n_removed
                        if 'layers_removed' in res and isinstance(res['layers_removed'], list):
                             if 'n_removed' in res:
                                 res['layers_removed'] = res['n_removed']
                             else:
                                 res['layers_removed'] = len(res['layers_removed'])

                        # Fix score key for TinyLlama
                        if 'score' not in res:
                            # ... score fixing logic stays the same but duplicated here for safety block context ...
                            if 'completion_accuracy' in res:
                                res['score'] = res['completion_accuracy']
                            elif 'accuracy' in res:
                                res['score'] = res['accuracy']
                
                # Merge logic: if model already exists, update it with new keys
                if model_name in results:
                    # Smart merge: keep lists if they are longer, etc.
                    # For now, just simplistic update of top-level keys
                    # If 'results' exists in both, prefer the one with more items
                    existing_data = results[model_name]
                    
                    if 'results' in data and 'results' in existing_data:
                         if len(data['results']) < len(existing_data['results']):
                             # Don't overwrite 'results' if new one is smaller/empty
                             del data['results']
                    
                    existing_data.update(data)
                else:
                    results[model_name] = data
                    
            except Exception as e:
                print(f"Warning: Failed to load {file}: {e}")
                
    return results


# ========== PLOT FUNCTIONS ==========

def plot_heatmap(
    model_name: str,
    bi_scores: Dict[str, float],
    output_path: Path,
    component: str = 'main'
) -> None:
    """Create horizontal bar heatmap of layer importance."""
    
    if not bi_scores:
        print(f"  Skipping heatmap for {model_name}: No BI scores available")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(bi_scores) * 0.25)))
    
    layers = sorted([int(k) for k in bi_scores.keys()])
    scores = [bi_scores[str(l)] for l in layers]
    
    # Color mapping
    colors = []
    for score in scores:
        if score < 0.1:
            colors.append('#4CAF50')  # Green - redundant
        elif score < 0.3:
            colors.append('#FFC107')  # Amber - moderate
        else:
            colors.append('#F44336')  # Red - important
    
    bars = ax.barh(layers, scores, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Value labels
    for i, (layer, score) in enumerate(zip(layers, scores)):
        ax.text(score + 0.02, layer, f'{score:.3f}', va='center', fontsize=9)
    
    ax.set_xlabel('Block Influence (BI)')
    ax.set_ylabel('Layer Index')
    ax.set_title(f'{model_name} - Layer Importance')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.set_yticks(layers)
    ax.invert_yaxis()
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#F44336', edgecolor='black', label='Critical (BI ≥ 0.3)'),
        mpatches.Patch(facecolor='#FFC107', edgecolor='black', label='Moderate (0.1 ≤ BI < 0.3)'),
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Redundant (BI < 0.1)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_line_graph(
    model_name: str,
    results_data: Dict,
    output_path: Path,
    metric_name: str = "Accuracy"
) -> None:
    """Create line graph for pruning sensitivity (Accuracy/Score vs Layers Removed)."""
    
    if 'results' not in results_data or not results_data['results']:
        print(f"  Skipping pruning plot for {model_name}: No benchmark results available")
        return

    # Extract data points
    configs = []
    scores = []
    for res in results_data['results']:
        if 'layers_removed' in res and 'score' in res:
            configs.append(int(res['layers_removed']))
            scores.append(res['score'])
            
    if not configs:
        print(f"  Skipping pruning plot for {model_name}: Valid data points not found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if reasoning model (heuristic) to pick color
    is_reasoning = "gsm8k" in str(output_path).lower() or "math" in model_name.lower() or "deepseek" in model_name.lower()
    color = '#9C27B0' if "math" in model_name.lower() else ('#2196F3' if is_reasoning else '#4CAF50')
    
    # Plot line
    ax.plot(configs, scores, marker='o', linewidth=3, markersize=10, 
             color=color, label=metric_name)
    
    # Fill area under line lightly
    ax.fill_between(configs, scores, alpha=0.1, color=color)

    # Add value labels
    for x, y in zip(configs, scores):
        ax.annotate(f'{y:.1f}%', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=11,
                    fontweight='bold')

    # Add baseline line
    baseline = scores[0]
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label='Baseline')

    # Styling
    ax.set_xlabel('Number of Layers Removed', fontsize=12)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=12)
    ax.set_title(f'{model_name} - Pruning Sensitivity', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(configs)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_bi_scores(
    model_name: str,
    bi_scores: Dict[str, float],
    output_path: Path
) -> None:
    """Create BI score curve (normalized position vs BI)."""
    
    if not bi_scores:
        # User requested to silence this
        # print(f"  Skipping bathtub curve for {model_name}: No BI scores available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = sorted([int(k) for k in bi_scores.keys()])
    scores = [bi_scores[str(l)] for l in layers]
    n_layers = len(layers)
    positions = [l / (n_layers - 1) for l in layers] if n_layers > 1 else [0.0]
    
    ax.plot(positions, scores, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax.fill_between(positions, scores, alpha=0.2, color='#2196F3')
    
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5, 
               label='Pruning threshold (0.1)')
    
    ax.set_xlabel('Normalized Layer Position (0=first, 1=last)')
    ax.set_ylabel('Block Influence (BI)')
    ax.set_title(f'{model_name} - Block Influence Scores')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, max(scores) * 1.1 if scores else 1.0)
    ax.legend(loc='upper right')
    
    # Annotate key layers
    for i, (pos, score) in enumerate(zip(positions, scores)):
        if score > 0.15 or i == 0 or i == n_layers - 1 or i == 2:
            ax.annotate(f'L{i}', (pos, score), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"Error saving BI scores plot: {e}")
        plt.close()


def plot_comparison(
    models_data: Dict[str, Dict],
    output_path: Path,
    title: str = "Model Comparison"
) -> None:
    """Plot multiple models on same curve for comparison."""
    
    valid_models = {name: data for name, data in models_data.items() 
                   if 'main' in data and 'bi_scores' in data['main'] and data['main']['bi_scores']}
    
    if len(valid_models) < 2:
        # User requested to silence this
        # print(f"  Skipping comparison plot {output_path}: Need at least 2 models with BI scores")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (model_name, data) in enumerate(valid_models.items()):
        bi_scores = data['main']['bi_scores']
        layers = sorted([int(k) for k in bi_scores.keys()])
        scores = [bi_scores[str(l)] for l in layers]
        n_layers = len(layers)
        positions = [l / (n_layers - 1) for l in layers] if n_layers > 1 else [0.0]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(positions, scores, f'{marker}-', linewidth=2, markersize=6,
                color=color, label=model_name, alpha=0.8)
    
    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2, 
               label='Pruning threshold')
    
    ax.set_xlabel('Normalized Layer Position')
    ax.set_ylabel('Block Influence (BI)')
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")
        plt.close()


def plot_summary_card(
    model_name: str,
    results: Dict,
    output_path: Path
) -> None:
    """Create infographic-style summary card."""
    
    if 'main' not in results or 'bi_scores' not in results['main'] or not results['main']['bi_scores']:
         print(f"  Skipping summary card for {model_name}: No BI scores available")
         return

    stats = results['main'].get('stats', {})
    bi_scores = results['main']['bi_scores']
    
    if not stats:
        # Compute stats if not present
        scores = list(bi_scores.values())
        redundant = [i for i, s in enumerate(scores) if s < 0.1]
        stats = {
            'total_layers': len(bi_scores),
            'redundant_count': len(redundant),
            'redundant_pct': len(redundant) / len(bi_scores) * 100 if scores else 0,
            'mean_bi': np.mean(scores) if scores else 0,
            'layer_2_bi': bi_scores.get('2', bi_scores.get(2, 0))
        }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    ax.text(0.5, 0.95, f'{model_name}', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    box_props = dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.5)
    
    ax.text(0.2, 0.7, f"Total Layers\n{stats['total_layers']}", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=box_props)
    
    ax.text(0.5, 0.7, f"Redundant\n{stats['redundant_count']} ({stats['redundant_pct']:.1f}%)", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.5))
    
    ax.text(0.8, 0.7, f"Mean BI\n{stats['mean_bi']:.4f}", 
            fontsize=16, ha='center', va='center', transform=ax.transAxes,
            bbox=box_props)
    
    ax.text(0.5, 0.4, f"Layer 2 BI: {stats['layer_2_bi']:.4f}", 
            fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    layer2_status = "HIGH" if stats['layer_2_bi'] > 0.2 else "LOW"
    ax.text(0.5, 0.3, f"Layer 2 Phenomenon: {layer2_status}", 
            fontsize=14, ha='center', va='center', transform=ax.transAxes,
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ========== MAIN GENERATORS ==========

def plot_benchmark_comparison(
    models_data: Dict[str, Dict],
    output_path: Path,
    metric_name: str = "Accuracy"
) -> None:
    """Plot pruning sensitivity comparison for multiple models."""
    
    valid_models = {name: data for name, data in models_data.items() 
                   if 'results' in data and data['results']}
    
    if len(valid_models) < 2:
        print(f"  Skipping benchmark comparison {output_path}: Need at least 2 models with results")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#795548', '#607D8B']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    
    for i, (model_name, data) in enumerate(valid_models.items()):
        configs = []
        scores = []
        for res in data['results']:
            if 'layers_removed' in res and 'score' in res:
                configs.append(int(res['layers_removed']))
                scores.append(res['score'])
        
        # Sort by layers removed
        points = sorted(zip(configs, scores))
        if not points:
            continue
            
        configs, scores = zip(*points)
        
        # Normalize relative to baseline for fair comparison if needed? 
        # For now, plotting raw accuracy/score as requested.
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(configs, scores, f'{marker}-', linewidth=2, markersize=8,
                color=color, label=model_name, alpha=0.8)
                
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Layers Removed')
    ax.set_ylabel(metric_name)
    ax.set_title("Benchmark Degradation Comparison")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"Error saving benchmark comparison: {e}")
        plt.close()


def generate_for_type(model_type: str, base_dir: Path = Path("results")) -> None:
    """Generate all visualizations for a model type."""
    
    data_dir = base_dir / "data" / model_type
    fig_dir = base_dir / "figures" / model_type
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading {model_type} results...")
    results = load_results(data_dir)
    
    if not results:
        print(f"  No results found in {data_dir}")
        return
    
    print(f"  Found {len(results)} model(s)")
    
    for model_name, data in results.items():
        print(f"\nGenerating figures for {model_name}...")
        safe_name = model_name.lower().replace('-', '_').replace('.', '_').replace('/', '_')
        
        # 1. Pruning Sensitivity Line Graph (if benchmark results exist)
        if 'results' in data and data['results']:
            plot_line_graph(model_name, data, fig_dir / f"{safe_name}_pruning.png")
        else:
             print(f"  Skipping sensitivity plot for {model_name}: No benchmark data")
        
        # 2. Block Influence Visualizations (if BI scores exist)
        if 'main' in data and 'bi_scores' in data['main'] and data['main']['bi_scores']:
            bi_scores = data['main']['bi_scores']
            
            # Heatmap
            plot_heatmap(model_name, bi_scores, 
                         fig_dir / f"{safe_name}_heatmap.png")
            
            # BI Scores curve (formerly bathtub)
            plot_bi_scores(model_name, bi_scores,
                         fig_dir / f"{safe_name}_bi_scores.png")
            
            # Summary card
            plot_summary_card(model_name, data,
                              fig_dir / f"{safe_name}_summary.png")
        else:
            # User requested to silence this
            # print(f"  Skipping BI plots for {model_name}: No BI scores")
            pass
            
    # Generate Type-Specific Comparison (if multiple models have BI scores)
    valid_bi_models = {name: d for name, d in results.items() if 'main' in d and 'bi_scores' in d['main']}
    if len(valid_bi_models) > 1:
        print(f"\nGenerating {model_type} BI comparison...")
        # Use filename requested by user: cross_model_bi_comparison.png
        plot_comparison(results, fig_dir / "cross_model_bi_comparison.png",
                       f"{model_type.title()} Models - BI Comparison")
                       
    # Generate Type-Specific Benchmark Comparison
    valid_bench_models = {name: d for name, d in results.items() if 'results' in d and d['results']}
    if len(valid_bench_models) > 1:
        print(f"\nGenerating {model_type} Benchmark comparison...")
        plot_benchmark_comparison(results, fig_dir / "benchmark_comparison.png",
                                "Task Performance (Accuracy/Perplexity)")


def generate_cross_comparison(base_dir: Path = Path("results")) -> None:
    """Generate comparison between language and reasoning models."""
    
    print("\nGenerating cross-model-type comparison...")
    
    fig_dir = base_dir / "figures" / "comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all needed data
    lang_results = load_results(base_dir / "data" / "language")
    reasoning_results = load_results(base_dir / "data" / "reasoning")
    
    # Find specific models if available, otherwise fallback
    target_lang = "Qwen2.5-1.5B-Instruct"
    target_reasoning = "Qwen2.5-Math-1.5B-Instruct"
    
    lang_model_name = target_lang if target_lang in lang_results else None
    reasoning_model_name = target_reasoning if target_reasoning in reasoning_results else None
    
    # Fallback to first available if target not found
    if not lang_model_name and lang_results:
        lang_model_name = list(lang_results.keys())[0]
        print(f"  Note: {target_lang} not found, using {lang_model_name} instead.")
        
    if not reasoning_model_name and reasoning_results:
        reasoning_model_name = list(reasoning_results.keys())[0]
        print(f"  Note: {target_reasoning} not found, using {reasoning_model_name} instead.")
        
    if not lang_model_name or not reasoning_model_name:
        print("  Skipping CoT vs LLM Comparison: Missing data for one or both categories.")
        return

    # Create the comparison
    combined = {
        f"General LLM ({lang_model_name})": lang_results[lang_model_name],
        f"Reasoning CoT ({reasoning_model_name})": reasoning_results[reasoning_model_name]
    }
    
    plot_comparison(
        combined, 
        fig_dir / "language_vs_reasoning_bi_comparison.png",
        "Layer Redundancy: General LLM vs Reasoning Specialized"
    )

    # Also check for DeepSeek
    ds_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    if ds_name in reasoning_results:
         combined_ds = {
             f"General LLM ({lang_model_name})": lang_results[lang_model_name],
             f"DeepSeek R1 ({ds_name})": reasoning_results[ds_name]
         }
         plot_comparison(
            combined_ds,
            fig_dir / "language_vs_deepseek_bi_comparison.png",
            "Layer Redundancy: General LLM vs DeepSeek R1"
         )


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="Visualization Generator")
    parser.add_argument('--type', choices=['language', 'reasoning', 'comparison', 'all'],
                        default='all', help="Type of visualizations to generate")
    args = parser.parse_args()
    
    base_dir = Path("results")
    
    if args.type in ['language', 'all']:
        generate_for_type('language', base_dir)
    
    if args.type in ['reasoning', 'all']:
        generate_for_type('reasoning', base_dir)
    
    if args.type in ['comparison', 'all']:
        generate_cross_comparison(base_dir)
    
    print("\n✅ Visualization generation complete!")


if __name__ == "__main__":
    main()
