"""
Custom Benchmark Template

This template demonstrates how to create custom benchmarks for the
Universal Model Pruning system. Copy this file and implement your
own benchmark following the interface.

Usage:
    1. Copy this file to experiments/benchmarks/your_benchmark.py
    2. Implement the required methods
    3. Register in experiments/benchmarks/__init__.py:
       from .your_benchmark import YourBenchmark
       BENCHMARKS['your_benchmark'] = YourBenchmark
    4. Run: python benchmark.py --model <model> --benchmark your_benchmark
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from .base import BaseBenchmark


class CustomBenchmark(BaseBenchmark):
    """
    Template for creating custom benchmarks.
    
    Required implementations:
    - load_dataset(): Load your evaluation dataset
    - evaluate(): Run model evaluation and return (score, details)
    - extract_answer(): Parse model output to extract answer
    """
    
    # Unique name for this benchmark (used in CLI)
    name: str = "custom"
    
    # Model type: 'language', 'vision', 'reasoning', 'audio'
    model_type: str = "language"
    
    def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load your evaluation dataset.
        
        Returns:
            List of sample dictionaries with at minimum:
            - 'input' or 'prompt': The input to the model
            - 'expected' or 'answer': The expected output
            
        Example:
            return [
                {'prompt': 'What is 2+2?', 'answer': '4'},
                {'prompt': 'Capital of France?', 'answer': 'Paris'},
            ]
        """
        # Replace with your dataset loading logic
        # Example using HuggingFace datasets:
        # from datasets import load_dataset
        # dataset = load_dataset("your_dataset_name")
        
        # Placeholder - replace with actual data
        samples = [
            {'prompt': 'Example prompt 1', 'answer': 'Expected answer 1'},
            {'prompt': 'Example prompt 2', 'answer': 'Expected answer 2'},
        ]
        
        if num_samples:
            samples = samples[:num_samples]
        
        return samples
    
    def evaluate(
        self,
        model,
        tokenizer,
        dataset: List[Dict],
        device: str = 'cuda',
        max_new_tokens: int = 100,
        greedy: bool = True
    ) -> Tuple[float, Dict]:
        """
        Evaluate model on the benchmark.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            dataset: Pre-loaded dataset from load_dataset()
            device: 'cuda' or 'cpu'
            max_new_tokens: Max tokens to generate
            greedy: Use greedy decoding
            
        Returns:
            Tuple of (accuracy_percentage, details_dict)
        """
        model.eval()
        correct = 0
        total = 0
        results = []
        
        pbar = tqdm(dataset, desc=f"{self.name} Eval", leave=False)
        for sample in pbar:
            prompt = sample.get('prompt') or sample.get('input')
            expected = self.get_expected_answer(sample)
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and compare answer
            predicted = self.extract_answer(response)
            is_correct = self.compare_answers(predicted, expected)
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                'prompt': prompt[:50],
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct
            })
            
            pbar.set_postfix({'acc': f'{correct/total*100:.1f}%'})
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        return accuracy, {
            'correct': correct,
            'total': total,
            'results': results[:10]  # Sample results
        }
    
    def extract_answer(self, response: str) -> Any:
        """
        Extract the answer from raw model response.
        
        Override this to implement custom answer extraction.
        
        Args:
            response: Full model output string
            
        Returns:
            Extracted answer (string, number, etc.)
        """
        # Simple extraction - take last line
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    # Optional: Override for custom answer format
    def get_expected_answer(self, sample: Dict) -> Any:
        """Extract expected answer from sample."""
        return sample.get('answer', sample.get('expected', ''))
    
    # Optional: Override for custom comparison
    def compare_answers(self, predicted: Any, expected: Any) -> bool:
        """Compare predicted vs expected. Default: case-insensitive."""
        if predicted is None:
            return False
        return str(predicted).strip().lower() == str(expected).strip().lower()
