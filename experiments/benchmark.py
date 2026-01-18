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

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
from src.handlers import UniversalHandler
from experiments.benchmarks import get_benchmark, list_benchmarks


# Vision model identifiers
VISION_MODEL_KEYWORDS = ['clip', 'siglip', 'vit', 'resnet', 'efficientnet', 'deit', 'swin', 'dinov2', 'dino', 'beit', 'jina']


def is_vision_model(model_id: str) -> bool:
    """Check if model_id is a vision model."""
    model_id_lower = model_id.lower()
    return any(kw in model_id_lower for kw in VISION_MODEL_KEYWORDS)


def load_model(model_id: str, device: str = 'cuda'):
    """Load model and tokenizer/processor from HuggingFace."""
    print(f"Loading model: {model_id}")
    
    if is_vision_model(model_id):
        return load_vision_model(model_id, device)
    else:
        return load_language_model(model_id, device)


def load_language_model(model_id: str, device: str = 'cuda'):
    """Load a language model (CausalLM) and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_vision_model(model_id: str, device: str = 'cuda'):
    """Load a vision model and processor."""
    from transformers import AutoModelForImageClassification, AutoProcessor, AutoModel
    
    # Check for models that need AutoModel (not classification head)
    # - Zero-shot models: CLIP, SigLIP, Jina-CLIP
    # - Feature extraction models: DINOv2, ViT (non-classification), Swin (non-classification)
    needs_automodel = any(x in model_id.lower() for x in ['clip', 'siglip', 'dinov2', 'dino', 'jina'])
    
    if needs_automodel:
        print("  Loading as Zero-Shot model (AutoModel)...")
        # Jina CLIP uses bfloat16 internally, others use float16
        if 'jina' in model_id.lower():
            dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        else:
            dtype = torch.float16 if device == 'cuda' else torch.float32
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device)
    else:
        print("  Loading as Classification model (AutoModelForImageClassification)...")
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device)
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except:
        # Fallback for models that use FeatureExtractor (like ResNet)
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor


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
    tolerance: float = 0.05,
    strategy: str = 'sequential'
) -> List[Dict]:
    """
    Automatically prune layers until performance degrades N times.
    
    Args:
        threshold: Number of consecutive degradations before stopping
        tolerance: Percentage drop considered a "degradation" (default 5%)
        strategy: 'sequential' (default) or 'bi' (Block Influence)
    
    Returns:
        List of results for each configuration tested
    """
    results = []
    degradations = 0
    layer_idx = 0
    baseline_score = None
    
    print(f"\nSmart Pruning Mode (strategy={strategy}, threshold={threshold}, tolerance={tolerance*100}%)")
    print("=" * 60)
    
    # Pre-calculate BI scores if needed
    pruning_order = []
    if strategy == 'bi':
        print("\nComputing Block Influence (BI) scores for pruning order...")
        # Load model for analysis
        model, tokenizer = load_model(model_id, device)
        handler = UniversalHandler(model, verbose=False)
        
        # Determine component
        components = handler.list_components()
        if 'main' in components:
            component_name = 'main'
        elif 'vision' in components:
            component_name = 'vision'
        elif 'encoder' in components:
            component_name = 'encoder'
        else:
            component_name = components[0] if components else 'main'
            
        print(f"  Analyzing component: {component_name}")
        
        # Create a simple dataloader for BI calculation
        # We assume dataset is iterable. For BI we need a cleaner loader usually, 
        # but UniversalHandler.compute_bi_scores handles raw dataloaders.
        # If dataset is a list/Dataset, wrap it.
        if not isinstance(dataset, torch.utils.data.DataLoader):
            # Handle list of strings (Language benchmark)
            if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], str):
                print("  Tokenizing text dataset for BI calculation...")
                # Limit samples for speed
                samples = dataset[:20]
                encodings = tokenizer(
                    samples,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )
                
                class TensorDataset(torch.utils.data.Dataset):
                    def __init__(self, encodings):
                        self.encodings = encodings
                    def __len__(self):
                        return len(self.encodings['input_ids'])
                    def __getitem__(self, idx):
                        return {k: v[idx] for k, v in self.encodings.items()}
                
                bi_loader = torch.utils.data.DataLoader(TensorDataset(encodings), batch_size=1)
            
            # Handle list of dicts (Vision benchmark)
            elif isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
                print("  Processing vision dataset for BI calculation...")
                samples = dataset[:20]  # Limit for speed
                
                class VisionDataset(torch.utils.data.Dataset):
                    def __init__(self, samples, processor):
                        self.samples = samples
                        self.processor = processor
                    
                    def __len__(self):
                        return len(self.samples)
                    
                    def __getitem__(self, idx):
                        sample = self.samples[idx]
                        image = sample['image']
                        # Process image using tokenizer (which is actually a processor for vision)
                        inputs = self.processor(images=image, return_tensors='pt')
                        # Remove batch dimension added by processor
                        return {k: v.squeeze(0) for k, v in inputs.items()}
                
                bi_loader = torch.utils.data.DataLoader(
                    VisionDataset(samples, tokenizer),
                    batch_size=1,
                    collate_fn=lambda batch: {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
                )
            else:
                # Fallback for datasets that might already emit tensors or dicts
                bi_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        else:
            bi_loader = dataset

        # Compute scores
        # Limit samples for speed
        bi_scores = handler.compute_bi_scores(bi_loader, component=component_name, num_samples=20)
        
        # Sort layers by BI (ascending = lowest/most redundant first)
        sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
        pruning_order = [idx for idx, score in sorted_layers]
        
        print(f"  BI Pruning Order (Lowest to Highest): {pruning_order}")
        
        # Cleanup
        del model, handler
        torch.cuda.empty_cache()
    
    while degradations < threshold:
        # Load fresh model for each config
        model, tokenizer = load_model(model_id, device)
        handler = UniversalHandler(model, verbose=False)
        
        # Auto-detect the primary component to prune
        components = handler.list_components()
        if 'main' in components:
            component_name = 'main'
        elif 'vision' in components:
            component_name = 'vision'
        elif 'encoder' in components:
            component_name = 'encoder'
        else:
            component_name = components[0] if components else 'main'
        
        n_layers = len(handler.get_layers(component_name))
        
        # Check if we've exceeded available layers
        if layer_idx >= n_layers - 1:  # Keep at least 1 layer
            print(f"  Reached maximum pruning depth ({layer_idx} layers)")
            break
        
        config_name = 'baseline' if layer_idx == 0 else f'{layer_idx}L_removed'
        print(f"\n[{config_name}]")
        
        # Remove layers (use inplace=True to modify the model directly)
        if layer_idx > 0:
            if strategy == 'bi':
                # Remove top N redundant layers based on BI order
                # pruning_order contains indices [least_important, ..., most_important]
                # We remove the first 'layer_idx' layers from this list
                layers_to_remove = pruning_order[:layer_idx]
            else:
                # Sequential: Remove layers 3, 4, 5... (standard bathtub assumption)
                layers_to_remove = list(range(3, 3 + layer_idx))
                
            handler.remove_layers(component_name, layers_to_remove, inplace=True)
            print(f"  Removed layers: {layers_to_remove}")
            print(f"  Model now has {len(handler.get_layers(component_name))} layers")
        
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
    # Include mode/strategy in filename to prevent overwriting
    safe_mode = mode.replace(' ', '_').lower()
    output_file = output_dir / f"{safe_name}_{benchmark_name}_{safe_mode}_benchmark.json"
    
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
    parser.add_argument('--prune-strategy', type=str, default='sequential',
                        choices=['sequential', 'bi'],
                        help='Pruning strategy: sequential (default) or bi (Block Influence)')
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
        mode = f'smart_prune_{args.prune_strategy}'
        results = smart_prune(
            args.model, benchmark, dataset, device,
            args.greedy, threshold=args.smart_prune,
            strategy=args.prune_strategy
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
