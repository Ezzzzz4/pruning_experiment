"""
Language Model Benchmark Plugin.

Tests language model quality via perplexity and generation coherence.
Designed for standard LLMs like GPT-2, TinyLlama, Qwen.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base import BaseBenchmark


class TextDataset(Dataset):
    """Simple text dataset for perplexity computation."""
    
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


class LanguageBenchmark(BaseBenchmark):
    """Language model benchmark using perplexity."""
    
    name = "language"
    model_type = "language"
    
    # Sample texts for evaluation
    SAMPLE_TEXTS = [
        "The capital of France is",
        "In the year 1969, humans first",
        "Machine learning is a field of",
        "The process of photosynthesis involves",
        "According to the theory of relativity,",
        "The main difference between DNA and RNA is",
        "Climate change is caused primarily by",
        "Artificial intelligence will likely",
        "The history of computing began with",
        "Quantum mechanics describes the behavior of",
    ]
    
    def load_dataset(self, num_samples: Optional[int] = None):
        """Generate sample texts for perplexity evaluation."""
        if num_samples is None:
            num_samples = 100
        
        # Repeat sample texts to reach desired count
        texts = self.SAMPLE_TEXTS * (num_samples // len(self.SAMPLE_TEXTS) + 1)
        return texts[:num_samples]
    
    def evaluate(
        self,
        model,
        tokenizer,
        dataset: List[str],
        device: str = 'cuda',
        max_new_tokens: int = 50,
        greedy: bool = True
    ) -> Tuple[float, Dict]:
        """
        Evaluate model via perplexity computation.
        
        Returns:
            (perplexity_score, {loss, num_tokens, examples})
            Lower perplexity = better, so we return 100 - log(ppl) as "score"
        """
        text_dataset = TextDataset(dataset, tokenizer, max_length=128)
        dataloader = DataLoader(text_dataset, batch_size=8, shuffle=False)
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        pbar = tqdm(dataloader, desc="Language", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.no_grad():
                labels = input_ids.clone()
                if tokenizer.pad_token_id is not None:
                    labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Update progress
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.3f}'})
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        # Convert to score (higher is better)
        # Score = 100 - log10(perplexity) * 20, capped at 0-100
        if perplexity < float('inf'):
            score = max(0, min(100, 100 - math.log10(perplexity) * 30))
        else:
            score = 0
        
        # Generate sample outputs for quality check
        examples = self._generate_examples(model, tokenizer, device, greedy)
        
        return score, {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'examples': examples
        }
    
    def _generate_examples(
        self, 
        model, 
        tokenizer, 
        device: str,
        greedy: bool,
        num_examples: int = 3
    ) -> List[Dict]:
        """Generate sample completions for quality review."""
        examples = []
        prompts = self.SAMPLE_TEXTS[:num_examples]
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                gen_kwargs = {
                    'max_new_tokens': 30,
                    'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                    'attention_mask': inputs['attention_mask']
                }
                
                if greedy:
                    gen_kwargs['do_sample'] = False
                    gen_kwargs['temperature'] = None
                    gen_kwargs['top_p'] = None
                else:
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = 0.7
                
                outputs = model.generate(inputs['input_ids'], **gen_kwargs)
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            examples.append({
                'prompt': prompt,
                'completion': completion
            })
        
        return examples
    
    def extract_answer(self, response: str) -> Any:
        """Not used for perplexity benchmark."""
        return response
    
    def compare_answers(self, predicted: Any, expected: Any) -> bool:
        """Not used for perplexity benchmark."""
        return True
