"""
Language Model Experiments - Week 2

Runs Block Influence analysis on language models:
- GPT-2 (baseline)
- TinyLlama-1.1B
- Qwen2.5-0.5B
- Qwen2.5-1.5B

Generates visualizations and saves results.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from src.handlers import UniversalHandler, create_handler
from src.core.block_influence import BlockInfluenceAnalyzer
from src.utils.visualization import PruningVisualizer
from src.utils.statistics import compute_stats_summary


class TextDataset(Dataset):
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
        return self.encodings['input_ids'].size(0)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_sample_texts(num_samples: int = 200) -> List[str]:
    """Generate sample texts for analysis."""
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Neural networks can learn complex patterns from data.",
        "Deep learning models require significant computational resources.",
        "Natural language processing enables computers to understand text.",
        "Artificial intelligence is advancing rapidly in recent years.",
        "The transformer architecture revolutionized NLP in 2017.",
        "Large language models can generate human-like text.",
        "Computer vision uses neural networks for image recognition.",
        "Reinforcement learning teaches agents through trial and error.",
        "The attention mechanism allows models to focus on relevant parts.",
        "Pre-training on large corpora improves downstream task performance.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Layer normalization helps stabilize training of deep networks.",
        "Residual connections enable training of very deep models.",
        "Dropout regularization prevents overfitting in neural networks.",
        "Batch size affects both training speed and model convergence.",
        "Learning rate scheduling can improve final model performance.",
        "Gradient clipping prevents exploding gradients during training.",
        "Weight decay regularization adds a penalty to large weights.",
    ]
    
    # Repeat and shuffle to get desired count
    texts = (base_texts * ((num_samples // len(base_texts)) + 1))[:num_samples]
    return texts


def analyze_model(
    model_name: str,
    model_id: str,
    output_dir: Path,
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict:
    """
    Run full BI analysis on a model.
    
    Returns dict with all results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}")
    
    # Load model and tokenizer
    print("\n[1/5] Loading model...")
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    # Create handler
    print("\n[2/5] Creating handler...")
    handler = UniversalHandler(model, verbose=True)
    
    # Create dataset
    print("\n[3/5] Preparing dataset...")
    texts = get_sample_texts(num_samples)
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Compute BI scores
    print("\n[4/5] Computing Block Influence scores...")
    results = {}
    
    for comp_name in handler.list_components():
        print(f"\n  Analyzing component: {comp_name}")
        bi_scores = handler.compute_bi_scores(dataloader, comp_name, num_samples)
        
        # Compute statistics
        scores = list(bi_scores.values())
        redundant = [idx for idx, s in bi_scores.items() if s < 0.1]
        
        results[comp_name] = {
            'bi_scores': bi_scores,
            'redundant_layers': redundant,
            'stats': {
                'total_layers': len(bi_scores),
                'redundant_count': len(redundant),
                'redundant_pct': len(redundant) / len(bi_scores) * 100,
                'mean_bi': float(np.mean(scores)),
                'std_bi': float(np.std(scores)),
                'min_bi': float(np.min(scores)),
                'max_bi': float(np.max(scores)),
                'layer_0_bi': float(bi_scores.get(0, 0)),
                'layer_1_bi': float(bi_scores.get(1, 0)),
                'layer_2_bi': float(bi_scores.get(2, 0)),
            }
        }
        
        # Print layer-by-layer results
        print(f"\n  Layer BI Scores:")
        for idx, score in sorted(bi_scores.items()):
            status = "🟢 REDUNDANT" if score < 0.1 else ("🟡 MODERATE" if score < 0.3 else "🔴 IMPORTANT")
            print(f"    Layer {idx:2d}: {score:.4f} {status}")
    
    # Add metadata
    results['metadata'] = {
        'model_name': model_name,
        'model_id': model_id,
        'num_samples': num_samples,
        'device': device,
        'timestamp': datetime.now().isoformat(),
        'is_multimodal': handler.is_multimodal,
        'components': handler.list_components(),
    }
    
    # Save results
    print("\n[5/5] Saving results...")
    results_file = output_dir / f"{model_name.lower().replace('-', '_')}_results.json"
    
    # Convert bi_scores keys to strings for JSON
    json_results = {}
    for key, value in results.items():
        if key == 'metadata':
            json_results[key] = value
        else:
            json_results[key] = {
                'bi_scores': {str(k): v for k, v in value['bi_scores'].items()},
                'redundant_layers': value['redundant_layers'],
                'stats': value['stats'],
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"  Saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    for comp_name, comp_results in results.items():
        if comp_name == 'metadata':
            continue
        stats = comp_results['stats']
        print(f"\n{comp_name}:")
        print(f"  Total layers: {stats['total_layers']}")
        print(f"  Redundant (BI<0.1): {stats['redundant_count']} ({stats['redundant_pct']:.1f}%)")
        print(f"  Mean BI: {stats['mean_bi']:.4f} ± {stats['std_bi']:.4f}")
        print(f"  Layer 2 BI: {stats['layer_2_bi']:.4f} (Layer 2 phenomenon check)")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    """Run all language model experiments."""
    
    # Configuration
    output_dir = Path("results/data/language")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Models to analyze
    models = [
        ("GPT2", "gpt2"),
        # ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        # ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
        # ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ]
    
    all_results = {}
    
    for model_name, model_id in models:
        try:
            results = analyze_model(
                model_name=model_name,
                model_id=model_id,
                output_dir=output_dir,
                num_samples=100,
                device=device
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n❌ Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    main()
