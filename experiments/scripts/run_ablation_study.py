"""
Layer Removal Ablation Study

Tests the effect of removing redundant layers on model performance:
1. Remove 1, 2, 4, 6, 8... layers progressively
2. Measure perplexity on held-out text
3. Measure inference speed
4. Generate comparison plots
"""

import os
import sys
import json
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from src.handlers import UniversalHandler


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 128
) -> float:
    """
    Compute perplexity on a set of texts.
    Lower perplexity = better language modeling.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            batch_texts = texts[i:i+4]
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Compute loss
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Count tokens (excluding padding)
            n_tokens = (inputs['input_ids'] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def measure_inference_speed(
    model,
    tokenizer,
    device: str = 'cuda',
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure inference speed (tokens/second).
    """
    model.eval()
    
    # Warmup
    input_text = "The quick brown fox"
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    for _ in range(3):
        with torch.no_grad():
            model(**inputs)
    
    # Time generation
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    tokens_generated = 50
    
    return {
        'avg_time_s': avg_time,
        'tokens_per_second': tokens_generated / avg_time,
        'std_time': np.std(times)
    }


def run_ablation(
    model_name: str,
    model_id: str,
    results_file: Path,
    output_dir: Path,
    device: str = 'cuda'
) -> Dict:
    """
    Run ablation study for a model.
    """
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {model_name}")
    print(f"{'='*60}")
    
    # Load BI results
    with open(results_file) as f:
        bi_results = json.load(f)
    
    bi_scores = bi_results['main']['bi_scores']
    bi_scores = {int(k): v for k, v in bi_scores.items()}
    
    # Sort layers by BI score (most redundant first)
    sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
    redundant_layers = [idx for idx, score in sorted_layers if score < 0.1]
    
    print(f"Found {len(redundant_layers)} redundant layers: {redundant_layers[:10]}...")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can understand complex patterns.",
        "Artificial intelligence is transforming many industries.",
        "Deep neural networks require significant computational resources.",
        "Natural language processing enables computers to understand text.",
    ] * 4  # 20 texts
    
    # Baseline measurements
    print("\n[Baseline] Measuring original model...")
    baseline_ppl = compute_perplexity(model, tokenizer, test_texts, device)
    baseline_speed = measure_inference_speed(model, tokenizer, device)
    
    print(f"  Perplexity: {baseline_ppl:.2f}")
    print(f"  Speed: {baseline_speed['tokens_per_second']:.1f} tok/s")
    
    # Ablation configurations
    n_layers = len(bi_scores)
    configs = [0, 1, 2, 4, 6, 8]
    configs = [c for c in configs if c <= len(redundant_layers)]
    
    results = {
        'model_name': model_name,
        'model_id': model_id,
        'total_layers': n_layers,
        'redundant_layers': redundant_layers,
        'baseline': {
            'perplexity': baseline_ppl,
            'speed': baseline_speed
        },
        'ablations': []
    }
    
    # Run ablations
    for n_remove in configs[1:]:  # Skip 0 (baseline)
        layers_to_remove = redundant_layers[:n_remove]
        
        print(f"\n[Ablation] Removing {n_remove} layers: {layers_to_remove}")
        
        # Create handler and remove layers
        handler = UniversalHandler(model, verbose=False)
        pruned_model = handler.remove_layers('main', layers_to_remove, inplace=False)
        pruned_model = pruned_model.to(device)
        
        # Measure
        try:
            ppl = compute_perplexity(pruned_model, tokenizer, test_texts, device)
            speed = measure_inference_speed(pruned_model, tokenizer, device)
            
            ppl_change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            speed_change = ((speed['tokens_per_second'] - baseline_speed['tokens_per_second']) / 
                           baseline_speed['tokens_per_second']) * 100
            
            print(f"  Perplexity: {ppl:.2f} ({ppl_change:+.1f}%)")
            print(f"  Speed: {speed['tokens_per_second']:.1f} tok/s ({speed_change:+.1f}%)")
            
            results['ablations'].append({
                'n_removed': n_remove,
                'layers_removed': layers_to_remove,
                'perplexity': ppl,
                'ppl_change_pct': ppl_change,
                'speed': speed,
                'speed_change_pct': speed_change
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['ablations'].append({
                'n_removed': n_remove,
                'layers_removed': layers_to_remove,
                'error': str(e)
            })
        
        # Clean up
        del pruned_model
        torch.cuda.empty_cache()
    
    # Save results
    output_file = output_dir / f"{model_name.lower().replace('.', '_').replace('-', '_')}_ablation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSaved to: {output_file}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    """Run ablation study for all language models."""
    
    output_dir = Path("results/data/language")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Models to test (only those with results)
    models = [
        ("GPT2", "gpt2", "results/data/language/gpt2_results.json"),
        ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "results/data/language/qwen2_5_0_5b_results.json"),
        ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "results/data/language/tinyllama_results.json"),
    ]
    
    all_results = {}
    
    for model_name, model_id, results_file in models:
        if not Path(results_file).exists():
            print(f"\n⚠️ Skipping {model_name}: No BI results found")
            continue
        
        try:
            results = run_ablation(
                model_name=model_name,
                model_id=model_id,
                results_file=Path(results_file),
                output_dir=output_dir,
                device=device
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
