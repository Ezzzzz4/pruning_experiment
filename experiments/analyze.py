"""
Unified Block Influence Analysis Script

Analyzes layer redundancy patterns across different model types:
- Language models (GPT-2, TinyLlama, Qwen2.5)
- Reasoning/CoT models (DeepSeek-R1-Distill)

Usage:
    python analyze.py --type language        # Standard LLMs
    python analyze.py --type reasoning       # CoT models
    python analyze.py --model MODEL_ID       # Specific model
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.handlers import UniversalHandler


# ========== MODEL REGISTRIES ==========

LANGUAGE_MODELS = [
    ("GPT-2", "gpt2"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]

REASONING_MODELS = [
    ("DeepSeek-R1-Distill-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    # "MobileLLM-R1-950M": Removed until verified
]


# ========== DATASETS ==========

class TextDataset(torch.utils.data.Dataset):
    """Simple text dataset for BI computation."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_sample_texts(model_type: str = "language", num_samples: int = 200) -> List[str]:
    """Generate sample texts appropriate for model type."""
    
    if model_type == "reasoning":
        # CoT-style prompts
        base_prompts = [
            "Let me think step by step. First,",
            "To solve this problem, I need to",
            "Breaking this down logically:",
            "Step 1: Analyze the problem.",
            "The key insight here is that",
            "We can solve this by first",
            "Thinking about this carefully,",
            "Let's work through this systematically.",
        ]
    else:
        # General language prompts
        base_prompts = [
            "The capital of France is",
            "In the year 1969, humans first",
            "Machine learning is a field of",
            "The process of photosynthesis involves",
            "According to the theory of relativity,",
            "The main difference between DNA and RNA is",
            "Climate change is caused primarily by",
            "Artificial intelligence will likely",
        ]
    
    # Repeat to get desired count
    texts = base_prompts * (num_samples // len(base_prompts) + 1)
    return texts[:num_samples]


# ========== ANALYSIS FUNCTIONS ==========

def analyze_model(
    model_name: str,
    model_id: str,
    model_type: str,
    output_dir: Path,
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict:
    """Run full BI analysis on a model."""
    
    print(f"\n{'='*70}")
    print(f"BI ANALYSIS: {model_name} ({model_type})")
    print(f"{'='*70}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create handler
    handler = UniversalHandler(model, verbose=True)
    print(f"Components discovered: {handler.list_components()}")
    
    # Create dataloader
    texts = get_sample_texts(model_type, num_samples)
    dataset = TextDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Compute BI scores
    print("\nComputing Block Influence scores...")
    bi_scores = handler.compute_bi_scores(dataloader, 'main', num_samples=num_samples)
    
    # Analyze results
    n_layers = len(bi_scores)
    redundant_layers = [i for i, score in bi_scores.items() if score < 0.1]
    scores_list = list(bi_scores.values())
    
    print(f"\n📊 Results:")
    print(f"  Total layers: {n_layers}")
    print(f"  Redundant (BI<0.1): {len(redundant_layers)} ({len(redundant_layers)/n_layers*100:.1f}%)")
    print(f"  Layer 2 BI: {bi_scores.get(2, 'N/A')}")
    print(f"  Mean BI: {np.mean(scores_list):.4f}")
    print(f"  Max BI: {max(scores_list):.4f}")
    print(f"  Min BI: {min(scores_list):.4f}")
    
    # Prepare results
    results = {
        'metadata': {
            'model_name': model_name,
            'model_id': model_id,
            'model_type': model_type,
            'num_samples': num_samples
        },
        'main': {
            'n_layers': n_layers,
            'bi_scores': {str(k): float(v) for k, v in bi_scores.items()},
            'stats': {
                'total_layers': n_layers,
                'redundant_count': len(redundant_layers),
                'redundant_pct': len(redundant_layers) / n_layers * 100,
                'mean_bi': float(np.mean(scores_list)),
                'max_bi': float(max(scores_list)),
                'min_bi': float(min(scores_list)),
                'layer_2_bi': float(bi_scores.get(2, 0))
            }
        }
    }
    
    # Save
    safe_name = model_name.lower().replace('-', '_').replace('.', '_')
    output_file = output_dir / f"{safe_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="Block Influence Analysis")
    parser.add_argument('--type', choices=['language', 'reasoning', 'all'], default='all',
                        help="Model type to analyze")
    parser.add_argument('--model', type=str, default=None,
                        help="Specific model ID to analyze")
    parser.add_argument('--samples', type=int, default=100,
                        help="Number of samples for BI computation")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Determine models to run
    models_to_run = []
    
    if args.model:
        # Single model mode
        models_to_run.append((args.model.split('/')[-1], args.model, 'custom'))
    else:
        if args.type in ['language', 'all']:
            for name, mid in LANGUAGE_MODELS:
                models_to_run.append((name, mid, 'language'))
        
        if args.type in ['reasoning', 'all']:
            for name, mid in REASONING_MODELS:
                models_to_run.append((name, mid, 'reasoning'))
    
    print(f"\nModels to analyze: {len(models_to_run)}")
    for name, _, mtype in models_to_run:
        print(f"  - {name} ({mtype})")
    
    all_results = {}
    
    for model_name, model_id, model_type in models_to_run:
        output_dir = Path(f"results/data/{model_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = analyze_model(
                model_name=model_name,
                model_id=model_id,
                model_type=model_type,
                output_dir=output_dir,
                num_samples=args.samples,
                device=device
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    if all_results:
        print("\n📊 Summary:")
        print("-" * 60)
        print(f"{'Model':<30} {'Layers':>8} {'Redundant':>12} {'L2 BI':>10}")
        print("-" * 60)
        for name, res in all_results.items():
            stats = res['main']['stats']
            print(f"{name:<30} {stats['total_layers']:>8} "
                  f"{stats['redundant_pct']:>10.1f}% "
                  f"{stats['layer_2_bi']:>10.3f}")


if __name__ == "__main__":
    main()
