"""
Abstract base class for all benchmark plugins.

Each benchmark must implement:
- name: str identifier
- model_type: 'reasoning', 'language', 'vision', 'audio'
- load_dataset(): Load the benchmark dataset
- evaluate(): Run evaluation and return score
- extract_answer(): Parse model output for scoring
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch


class BaseBenchmark(ABC):
    """Abstract base class for benchmark plugins."""
    
    name: str = "base"
    model_type: str = "unknown"  # reasoning, language, vision, audio
    
    @abstractmethod
    def load_dataset(self, num_samples: Optional[int] = None):
        """
        Load the benchmark dataset.
        
        Args:
            num_samples: Optional limit on samples. None = full dataset.
        
        Returns:
            Dataset or list of samples
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        dataset,
        device: str = 'cuda',
        max_new_tokens: int = 1024,
        greedy: bool = True
    ) -> Tuple[float, Dict]:
        """
        Evaluate model on the benchmark.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            dataset: Pre-loaded dataset
            device: cuda or cpu
            max_new_tokens: Generation limit
            greedy: Use greedy decoding (deterministic)
        
        Returns:
            Tuple of (accuracy_score, detailed_results_dict)
        """
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """
        Extract the answer from model response.
        
        Args:
            response: Raw model output string
        
        Returns:
            Extracted answer (type depends on benchmark)
        """
        pass
    
    def get_expected_answer(self, sample: Dict) -> Any:
        """
        Extract expected answer from dataset sample.
        Override if dataset format differs.
        """
        return sample.get('answer', sample.get('expected'))
    
    def compare_answers(self, predicted: Any, expected: Any) -> bool:
        """
        Compare predicted vs expected answer.
        Override for custom comparison logic.
        """
        if predicted is None:
            return False
        
        # Try numeric comparison
        try:
            return abs(float(predicted) - float(expected)) < 1e-6
        except (ValueError, TypeError):
            pass
        
        # String comparison
        return str(predicted).strip().lower() == str(expected).strip().lower()
