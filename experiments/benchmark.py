"""
Universal Model Pruning Benchmark CLI

Unified benchmark system supporting multiple model types and benchmarks.
Includes smart pruning to automatically find optimal layer removal.

Usage:
    # Manual layer selection
    python benchmark.py --model gpt2 --benchmark language --layers 1 2 4
    
    # Smart pruning (stop after N degradations)
    python benchmark.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                        --benchmark gsm8k --smart-prune 4
    
    # Full options
    python benchmark.py --model <HF_ID> \
                        --benchmark <gsm8k|language> \
                        --samples 50 \
                        --layers 1 2 4 8 \
                        --smart-prune 4 \
                        --greedy \
                        --no-graph \
                        --output results/data/
"""

import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.handlers import UniversalHandler
from experiments.benchmarks import get_benchmark, list_benchmarks


def load_model(model_id: str, device: str = 'cuda'):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def run_benchmark_config(
    model,
    tokenizer,
    handler: UniversalHandler,
    benchmark,
    dataset,
    n_remove: int,
    device: str,
    greedy: bool
) -> Tuple[float, Dict]:
    """Run benchmark for a specific layer removal configuration."""
    
    # Remove layers if needed
    if n_remove > 0:
        # Start from layer 3 (skip embedding layers)
        layers_to_remove = list(range(3, 3 + n_remove))
        handler.remove_layers('main', layers_to_remove)
        print(f"  Removed layers: {layers_to_remove}")
    
    # Evaluate
    score, details = benchmark.evaluate(
        model, tokenizer, dataset,
        device=device, greedy=greedy
    )
    
    return score, details


def smart_prune(
    model_id: str,
    benchmark,
    dataset,
    device: str,
    greedy: bool,
    threshold: int = 4,
    tolerance: float = 0.05
) -> List[Dict]:
    """
    Automatically prune layers until performance degrades N times.
    
    Args:
        threshold: Number of consecutive degradations before stopping
        tolerance: Percentage drop considered a "degradation" (default 5%)
    
    Returns:
        List of results for each configuration tested
    """
    results = []
    degradations = 0
    layer_idx = 0
    baseline_score = None
    
    print(f"\nSmart Pruning Mode (threshold={threshold}, tolerance={tolerance*100}%)")
    print("=" * 60)
    
    while degradations < threshold:
        # Load fresh model for each config
        model, tokenizer = load_model(model_id, device)
        handler = UniversalHandler(model, verbose=False)
        n_layers = len(handler.get_layers('main'))
        
        # Check if we've exceeded available layers
        if layer_idx >= n_layers - 3:  # Keep at least 3 layers
            print(f"  Reached maximum pruning depth ({layer_idx} layers)")
            break
        
        config_name = 'baseline' if layer_idx == 0 else f'{layer_idx}L_removed'
        print(f"\n[{config_name}]")
        
        # Remove layers (use inplace=True to modify the model directly)
        if layer_idx > 0:
            layers_to_remove = list(range(3, 3 + layer_idx))
            handler.remove_layers('main', layers_to_remove, inplace=True)
            print(f"  Removed layers: {layers_to_remove}")
            print(f"  Model now has {len(handler.get_layers('main'))} layers")
        
        # Evaluate
        score, details = benchmark.evaluate(
            model, tokenizer, dataset,
            device=device, greedy=greedy
        )
        
        print(f"  Score: {score:.2f}")
        
        # Track degradation
        if baseline_score is None:
            baseline_score = score
        elif score < baseline_score * (1 - tolerance):
            degradations += 1
            print(f"  ⚠️ Degradation #{degradations}")
        else:
            degradations = 0  # Reset on recovery
        
        results.append({
            'config': config_name,
            'layers_removed': layer_idx,
            'score': score,
            'details': details,
            'degradation': degradations > 0
        })
        
        # Cleanup
        del model, handler
        torch.cuda.empty_cache()
        
        layer_idx += 1
    
    print(f"\nStopped after {threshold} consecutive degradations")
    return results


def manual_prune(
    model_id: str,
    benchmark,
    dataset,
    device: str,
    greedy: bool,
    layer_counts: List[int]
) -> List[Dict]:
    """Run benchmark for specific layer removal counts."""
    results = []
    
    print(f"\nManual Pruning Mode (layers: {layer_counts})")
    print("=" * 60)
    
    for n_remove in [0] + layer_counts:
        config_name = 'baseline' if n_remove == 0 else f'{n_remove}L_removed'
        print(f"\n[{config_name}]")
        
        # Load fresh model
        model, tokenizer = load_model(model_id, device)
        handler = UniversalHandler(model, verbose=False)
        
        # Remove layers (use inplace=True to modify the model directly)
        if n_remove > 0:
            layers_to_remove = list(range(3, 3 + n_remove))
            handler.remove_layers('main', layers_to_remove, inplace=True)
            print(f"  Removed layers: {layers_to_remove}")
            print(f"  Model now has {len(handler.get_layers('main'))} layers")
        
        # Evaluate
        score, details = benchmark.evaluate(
            model, tokenizer, dataset,
            device=device, greedy=greedy
        )
        
        print(f"  Score: {score:.2f}")
        
        results.append({
            'config': config_name,
            'layers_removed': n_remove,
            'score': score,
            'details': details
        })
        
        # Cleanup
        del model, handler
        torch.cuda.empty_cache()
    
    return results


def generate_graph(results: List[Dict], model_name: str, benchmark_name: str, output_dir: Path):
    """Generate accuracy vs layers removed graph."""
    layers = [r['layers_removed'] for r in results]
    scores = [r['score'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, scores, 'b-o', linewidth=2, markersize=8)
    
    # Highlight baseline
    plt.axhline(y=scores[0], color='g', linestyle='--', alpha=0.5, label='Baseline')
    
    # Highlight degradation threshold (5% below baseline)
    threshold = scores[0] * 0.95
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='5% Threshold')
    
    plt.xlabel('Layers Removed', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'{model_name} - {benchmark_name.upper()} Benchmark', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    safe_name = model_name.replace('/', '_').replace('-', '_').lower()
    output_file = output_dir / f"{safe_name}_{benchmark_name}_pruning.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Graph saved: {output_file}")


def save_results(results: List[Dict], model_id: str, benchmark_name: str, mode: str, output_dir: Path):
    """Save benchmark results to JSON."""
    model_name = model_id.split('/')[-1]
    
    output_data = {
        'model_name': model_name,
        'model_id': model_id,
        'benchmark': benchmark_name,
        'mode': mode,
        'generated_at': datetime.now().isoformat(),
        'results': results
    }
    
    safe_name = model_name.replace('-', '_').replace('.', '_').lower()
    output_file = output_dir / f"{safe_name}_{benchmark_name}_benchmark.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📄 Results saved: {output_file}")


def print_summary(results: List[Dict]):
    """Print results summary table."""
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Config':<20} {'Score':<15} {'Layers Removed'}")
    print("-" * 50)
    for r in results:
        print(f"{r['config']:<20} {r['score']:.2f}{'%' if r['score'] <= 100 else '':<10} {r['layers_removed']}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Model Pruning Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --model gpt2 --benchmark language --layers 1 2 4
  python benchmark.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --benchmark gsm8k --smart-prune 4
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=list_benchmarks(),
                        help=f'Benchmark type: {", ".join(list_benchmarks())}')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--layers', nargs='+', type=int,
                        help='Specific layer counts to remove (e.g., 1 2 4 8)')
    parser.add_argument('--smart-prune', type=int, metavar='N',
                        help='Auto-prune until N consecutive degradations')
    parser.add_argument('--greedy', action='store_true', default=True,
                        help='Use greedy decoding (default: True)')
    parser.add_argument('--no-greedy', action='store_false', dest='greedy',
                        help='Use sampling instead of greedy')
    parser.add_argument('--no-graph', action='store_true',
                        help='Disable automatic graph generation')
    parser.add_argument('--output', type=str, default='results/data',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate args
    if not args.layers and not args.smart_prune:
        parser.error("Must specify either --layers or --smart-prune")
    
    if args.layers and args.smart_prune:
        parser.error("Cannot use both --layers and --smart-prune")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Get benchmark first to determine output subdirectory
    benchmark = get_benchmark(args.benchmark)
    print(f"Benchmark: {benchmark.name} ({benchmark.model_type})")
    
    # Output directory based on model type (reasoning vs language)
    base_output = Path(args.output)
    output_dir = base_output / benchmark.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Load dataset
    print(f"Loading dataset ({args.samples} samples)...")
    dataset = benchmark.load_dataset(args.samples)
    
    # Run benchmark
    model_name = args.model.split('/')[-1]
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*60}")
    
    if args.smart_prune:
        mode = 'smart_prune'
        results = smart_prune(
            args.model, benchmark, dataset, device,
            args.greedy, threshold=args.smart_prune
        )
    else:
        mode = 'manual'
        results = manual_prune(
            args.model, benchmark, dataset, device,
            args.greedy, args.layers
        )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.model, args.benchmark, mode, output_dir)
    
    # Generate graph (unless disabled)
    if not args.no_graph and len(results) > 1:
        generate_graph(results, model_name, args.benchmark, figures_dir)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
