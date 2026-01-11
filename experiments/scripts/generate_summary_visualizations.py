"""
Generate comprehensive visualizations for Language Model experiments.

Creates:
1. Cross-model comparison charts
2. Ablation study plots (perplexity + benchmark)
3. Bathtub curves (all models overlaid)
4. Pareto frontier (speed vs quality)
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
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

# Color palette for models
MODEL_COLORS = {
    'GPT2': '#2196F3',
    'Qwen2.5-0.5B': '#4CAF50', 
    'TinyLlama': '#FF9800'
}


def load_all_results(data_dir: Path) -> Dict:
    """Load all result files."""
    results = {}
    for file in data_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            # Extract model name from metadata or filename
            if 'metadata' in data:
                name = data['metadata'].get('model_name', file.stem)
            elif 'model_name' in data:
                name = data['model_name']
            else:
                name = file.stem
            results[file.stem] = data
    return results


def plot_cross_model_bi_comparison(results: Dict, output_path: Path):
    """Compare BI patterns across all models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models_data = {}
    for key, data in results.items():
        if 'results' in key and 'main' in data:
            name = data['metadata']['model_name']
            bi_scores = {int(k): v for k, v in data['main']['bi_scores'].items()}
            # Normalize to 0-1 position
            n_layers = len(bi_scores)
            positions = [i / (n_layers - 1) for i in range(n_layers)]
            scores = [bi_scores[i] for i in range(n_layers)]
            models_data[name] = (positions, scores)
    
    for name, (positions, scores) in models_data.items():
        color = MODEL_COLORS.get(name, '#666666')
        ax.plot(positions, scores, 'o-', label=name, color=color, 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, 
               label='Redundancy threshold (0.1)')
    
    ax.set_xlabel('Normalized Layer Position (0=first, 1=last)')
    ax.set_ylabel('Block Influence (BI)')
    ax.set_title('Cross-Model Layer Importance Comparison\n(Lower BI = More Redundant)')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_benchmark_comparison(results: Dict, output_path: Path):
    """Compare benchmark results across models and pruning levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['completion_accuracy', 'hellaswag_accuracy', 'coherence_score']
    titles = ['Text Completion Accuracy', 'HellaSwag Commonsense', 'Generation Coherence']
    
    benchmark_data = {}
    for key, data in results.items():
        if 'benchmark' in key and 'configurations' in data:
            name = data['model_name']
            benchmark_data[name] = data['configurations']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        x_labels = ['Baseline', '1 Layer', '2 Layers', '4 Layers']
        x = np.arange(len(x_labels))
        width = 0.25
        
        for i, (name, configs) in enumerate(benchmark_data.items()):
            values = []
            for config in configs:
                if metric in config:
                    values.append(config[metric])
                else:
                    values.append(0)
            
            color = MODEL_COLORS.get(name, '#666666')
            offset = (i - 1) * width
            ax.bar(x + offset, values[:len(x_labels)], width, label=name, color=color, alpha=0.8)
        
        ax.set_xlabel('Layers Removed')
        ax.set_ylabel('Score' if 'coherence' in metric else 'Accuracy (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_perplexity(results: Dict, output_path: Path):
    """Plot perplexity change across ablation levels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ablation_data = {}
    for key, data in results.items():
        if 'ablation' in key:
            # Extract model name from key
            model_key = key.replace('_ablation', '')
            if 'model_name' in data:
                name = data['model_name']
            else:
                name = model_key.replace('_', '-').title()
            
            if 'baseline' in data:
                baseline_ppl = data['baseline']['perplexity']
                ablation_data[name] = {'baseline': baseline_ppl, 'ablations': data.get('ablations', [])}
            elif 'configurations' in data:
                # Handle alternative format
                continue
    
    x = [0, 1, 2, 4, 6, 8]
    
    for name, data in ablation_data.items():
        baseline = data['baseline']
        ppl_values = [baseline]
        
        for abl in data['ablations']:
            if 'perplexity' in abl:
                ppl_values.append(abl['perplexity'])
        
        # Normalize relative to baseline
        ppl_normalized = [p / baseline * 100 for p in ppl_values]
        
        color = MODEL_COLORS.get(name, '#666666')
        ax.plot(x[:len(ppl_normalized)], ppl_normalized, 'o-', label=name, 
                color=color, linewidth=2, markersize=8)
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=110, color='orange', linestyle=':', label='10% degradation')
    ax.axhline(y=150, color='red', linestyle=':', label='50% degradation')
    
    ax.set_xlabel('Number of Layers Removed')
    ax.set_ylabel('Perplexity (% of baseline)')
    ax.set_title('Perplexity Degradation by Layer Removal')
    ax.legend(loc='upper left')
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_recommendation_summary(output_path: Path):
    """Create a summary recommendation chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['GPT-2', 'Qwen2.5-0.5B', 'TinyLlama']
    safe_layers = [2, 1, 2]
    total_layers = [12, 24, 22]
    pct_safe = [s/t*100 for s, t in zip(safe_layers, total_layers)]
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    bars = ax.barh(models, pct_safe, color=colors, alpha=0.8, edgecolor='black')
    
    # Add labels
    for i, (bar, safe, total) in enumerate(zip(bars, safe_layers, total_layers)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{safe}/{total} layers', va='center', fontsize=11)
    
    ax.set_xlabel('% of Layers Safely Removable')
    ax.set_title('Recommended Layer Pruning by Model\n(Based on Benchmark Tests)')
    ax.set_xlim(0, 25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations():
    """Generate all visualizations."""
    data_dir = Path("results/data/language")
    output_dir = Path("results/figures/language")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(data_dir)
    print(f"Found {len(results)} result files")
    
    print("\nGenerating visualizations...")
    
    # 1. Cross-model BI comparison
    plot_cross_model_bi_comparison(results, output_dir / "cross_model_bi_comparison.png")
    
    # 2. Benchmark comparison
    plot_benchmark_comparison(results, output_dir / "benchmark_comparison.png")
    
    # 3. Perplexity ablation
    plot_ablation_perplexity(results, output_dir / "perplexity_ablation.png")
    
    # 4. Recommendation summary
    plot_recommendation_summary(output_dir / "pruning_recommendations.png")
    
    # 5. Regenerate per-model visualizations
    from generate_visualizations import generate_all_visualizations as gen_per_model
    gen_per_model()
    
    print("\n✅ All visualizations generated!")


if __name__ == "__main__":
    generate_all_visualizations()
