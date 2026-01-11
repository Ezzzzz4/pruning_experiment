"""
Real Benchmark Evaluation for Pruned Language Models

Tests models on actual NLP tasks instead of just perplexity:
1. Text Completion Accuracy - Can the model complete sentences correctly?
2. Next Word Prediction - Accuracy on predicting next tokens
3. Generation Coherence - Does generated text make sense?
4. HellaSwag-style - Commonsense reasoning (simplified subset)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.handlers import UniversalHandler


# ============================================================
# BENCHMARK DATASETS
# ============================================================

COMPLETION_TESTS = [
    # (prompt, expected_continuation_keywords)
    ("The capital of France is", ["Paris"]),
    ("Water freezes at", ["0", "32", "zero", "degrees"]),
    ("The sun rises in the", ["east", "morning"]),
    ("Two plus two equals", ["four", "4"]),
    ("The largest planet in our solar system is", ["Jupiter"]),
    ("Dogs are known for their", ["loyalty", "bark", "smell", "hearing"]),
    ("The opposite of hot is", ["cold"]),
    ("Humans need oxygen to", ["breathe", "live", "survive"]),
    ("The color of the sky on a clear day is", ["blue"]),
    ("Einstein is famous for his theory of", ["relativity"]),
]

HELLASWAG_STYLE = [
    # (context, correct_answer_idx, choices)
    {
        "context": "A person is making a sandwich. They put bread on a plate and",
        "choices": [
            "started dancing in the kitchen",
            "added some cheese and lettuce",
            "threw the bread out the window",
            "called the police immediately"
        ],
        "correct": 1
    },
    {
        "context": "The student opened their textbook to study for the exam. They",
        "choices": [
            "ate the book for breakfast",
            "started reading the chapter carefully",
            "flew to the moon on a rocket",
            "turned into a dinosaur"
        ],
        "correct": 1
    },
    {
        "context": "It started raining heavily outside. The person grabbed",
        "choices": [
            "a sunscreen and sunglasses",
            "an umbrella before going out",
            "a swimming pool",
            "the clouds from the sky"
        ],
        "correct": 1
    },
    {
        "context": "The chef was preparing dinner in the kitchen. They chopped vegetables and",
        "choices": [
            "put them in a pot on the stove",
            "uploaded them to the internet",
            "fed them to invisible dragons",
            "planted them in the ceiling"
        ],
        "correct": 0
    },
    {
        "context": "A baby was crying because it was hungry. The mother",
        "choices": [
            "taught it quantum physics",
            "gave it a bottle of milk",
            "asked it to do taxes",
            "sent it to another dimension"
        ],
        "correct": 1
    },
]

COHERENCE_PROMPTS = [
    "Once upon a time, there was a",
    "The scientist discovered that",
    "In the future, technology will",
    "The most important thing in life is",
    "When you make a mistake, you should",
]


def evaluate_completion_accuracy(
    model,
    tokenizer,
    device: str = 'cuda',
    tests: List[Tuple] = COMPLETION_TESTS
) -> Dict:
    """
    Test if model generates text containing expected keywords.
    """
    model.eval()
    correct = 0
    results = []
    
    for prompt, expected_keywords in tests:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].lower()
        
        is_correct = any(kw.lower() in continuation for kw in expected_keywords)
        if is_correct:
            correct += 1
        
        results.append({
            'prompt': prompt,
            'generated': continuation.strip()[:50],
            'expected': expected_keywords,
            'correct': is_correct
        })
    
    return {
        'accuracy': correct / len(tests) * 100,
        'correct': correct,
        'total': len(tests),
        'details': results
    }


def evaluate_hellaswag_style(
    model,
    tokenizer,
    device: str = 'cuda',
    tests: List[Dict] = HELLASWAG_STYLE
) -> Dict:
    """
    HellaSwag-style commonsense reasoning:
    Choose the most likely continuation from multiple choices.
    """
    model.eval()
    correct = 0
    results = []
    
    for test in tests:
        context = test['context']
        choices = test['choices']
        correct_idx = test['correct']
        
        # Score each choice by computing likelihood
        scores = []
        for choice in choices:
            full_text = context + " " + choice
            inputs = tokenizer(full_text, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
            
            scores.append(-loss)  # Higher score = lower loss = more likely
        
        predicted_idx = np.argmax(scores)
        is_correct = predicted_idx == correct_idx
        
        if is_correct:
            correct += 1
        
        results.append({
            'context': context[:50] + "...",
            'predicted': choices[predicted_idx][:30],
            'correct': choices[correct_idx][:30],
            'is_correct': is_correct
        })
    
    return {
        'accuracy': correct / len(tests) * 100,
        'correct': correct,
        'total': len(tests),
        'details': results
    }


def evaluate_generation_coherence(
    model,
    tokenizer,
    device: str = 'cuda',
    prompts: List[str] = COHERENCE_PROMPTS,
    max_tokens: int = 50
) -> Dict:
    """
    Generate text and compute a coherence score based on:
    1. Repetition penalty (do words repeat too much?)
    2. Length (did it generate reasonable length?)
    3. Valid tokens (no garbage characters?)
    """
    model.eval()
    generations = []
    total_score = 0
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):]
        
        # Compute coherence metrics
        words = continuation.split()
        
        # 1. Repetition score (penalize repeated words)
        unique_words = len(set(words))
        total_words = len(words) if words else 1
        repetition_score = unique_words / total_words
        
        # 2. Length score (reward reasonable length)
        length_score = min(len(words) / 10, 1.0)  # Max at 10 words
        
        # 3. Valid characters (penalize garbage)
        valid_chars = sum(1 for c in continuation if c.isalnum() or c.isspace() or c in '.,!?')
        total_chars = len(continuation) if continuation else 1
        valid_score = valid_chars / total_chars
        
        coherence = (repetition_score + length_score + valid_score) / 3 * 100
        total_score += coherence
        
        generations.append({
            'prompt': prompt,
            'generated': continuation[:100],
            'coherence_score': coherence
        })
    
    return {
        'avg_coherence': total_score / len(prompts),
        'generations': generations
    }


def run_full_benchmark(
    model_name: str,
    model_id: str,
    bi_results_file: Path,
    output_dir: Path,
    device: str = 'cuda',
    n_layers_to_test: List[int] = [0, 1, 2, 4]
) -> Dict:
    """
    Run full benchmark suite on a model with various pruning levels.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {model_name}")
    print(f"{'='*60}")
    
    # Load BI results
    with open(bi_results_file) as f:
        bi_results = json.load(f)
    
    bi_scores = {int(k): v for k, v in bi_results['main']['bi_scores'].items()}
    sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
    redundant_layers = [idx for idx, score in sorted_layers if score < 0.1]
    
    print(f"Redundant layers: {redundant_layers[:8]}...")
    
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
    
    all_results = {'model_name': model_name, 'configurations': []}
    
    for n_remove in n_layers_to_test:
        if n_remove > len(redundant_layers):
            continue
        
        config_name = f"{n_remove}_layers_removed" if n_remove > 0 else "baseline"
        print(f"\n[{config_name}]")
        
        # Prune model if needed
        if n_remove > 0:
            handler = UniversalHandler(model, verbose=False)
            test_model = handler.remove_layers('main', redundant_layers[:n_remove], inplace=False)
            test_model = test_model.to(device)
        else:
            test_model = model
        
        # Run benchmarks
        print("  Running completion test...")
        completion = evaluate_completion_accuracy(test_model, tokenizer, device)
        
        print("  Running HellaSwag-style test...")
        hellaswag = evaluate_hellaswag_style(test_model, tokenizer, device)
        
        print("  Running coherence test...")
        coherence = evaluate_generation_coherence(test_model, tokenizer, device)
        
        result = {
            'config': config_name,
            'n_removed': n_remove,
            'layers_removed': redundant_layers[:n_remove] if n_remove > 0 else [],
            'completion_accuracy': completion['accuracy'],
            'hellaswag_accuracy': hellaswag['accuracy'],
            'coherence_score': coherence['avg_coherence'],
            'details': {
                'completion': completion,
                'hellaswag': hellaswag,
                'coherence': coherence
            }
        }
        
        all_results['configurations'].append(result)
        
        print(f"  Completion: {completion['accuracy']:.1f}%")
        print(f"  HellaSwag: {hellaswag['accuracy']:.1f}%")
        print(f"  Coherence: {coherence['avg_coherence']:.1f}")
        
        # Cleanup if we created a pruned model
        if n_remove > 0:
            del test_model
            torch.cuda.empty_cache()
    
    # Save results
    output_file = output_dir / f"{model_name.lower().replace('.', '_').replace('-', '_')}_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nSaved to: {output_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return all_results


def main():
    """Run benchmarks on all models."""
    output_dir = Path("results/data/language")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models = [
        ("GPT2", "gpt2", "results/data/language/gpt2_results.json"),
        ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "results/data/language/qwen2_5_0_5b_results.json"),
        ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "results/data/language/tinyllama_results.json"),
    ]
    
    for model_name, model_id, results_file in models:
        if not Path(results_file).exists():
            print(f"Skipping {model_name}: No BI results")
            continue
        
        try:
            run_full_benchmark(
                model_name=model_name,
                model_id=model_id,
                bi_results_file=Path(results_file),
                output_dir=output_dir,
                device=device
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("BENCHMARKS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
