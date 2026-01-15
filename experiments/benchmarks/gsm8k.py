"""
GSM8K Benchmark Plugin for Reasoning Models.

Tests mathematical reasoning ability on grade-school math word problems.
Designed for Chain-of-Thought (CoT) models like DeepSeek-R1.
"""

import re
from typing import Any, Dict, Optional, Tuple
import torch
from tqdm import tqdm
from datasets import load_dataset

from .base import BaseBenchmark


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K benchmark for mathematical reasoning."""
    
    name = "gsm8k"
    model_type = "reasoning"
    
    def load_dataset(self, num_samples: Optional[int] = None):
        """Load GSM8K test split."""
        dataset = load_dataset("gsm8k", "main", split="test")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        return dataset
    
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
        Evaluate model on GSM8K.
        
        Returns:
            (accuracy_percentage, {correct, total, examples})
        """
        correct = 0
        total = 0
        examples = []
        
        pbar = tqdm(dataset, desc="GSM8K", leave=False)
        for i, item in enumerate(pbar):
            question = item['question']
            expected = self.get_expected_answer(item)
            
            # Format with chat template
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Generate
            with torch.no_grad():
                gen_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'pad_token_id': tokenizer.pad_token_id,
                    'attention_mask': inputs['attention_mask']
                }
                
                if greedy:
                    gen_kwargs['do_sample'] = False
                    # Remove sampling parameters if they exist in config defaults
                    gen_kwargs['temperature'] = None
                    gen_kwargs['top_p'] = None
                    gen_kwargs['top_k'] = None
                else:
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = 0.6
                
                outputs = model.generate(inputs['input_ids'], **gen_kwargs)
            
            # Decode new tokens only
            new_tokens = outputs[0][len(inputs['input_ids'][0]):]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            predicted = self.extract_answer(response)
            is_correct = self.compare_answers(predicted, expected)
            
            if is_correct:
                correct += 1
            total += 1
            
            # Update progress
            accuracy = (correct / total) * 100
            pbar.set_postfix({'acc': f'{accuracy:.1f}%'})
            
            # Store examples (first 10 + failures)
            if len(examples) < 10 or (not is_correct and len(examples) < 20):
                examples.append({
                    'question': question[:100] + '...',
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct
                })
        
        accuracy = (correct / total * 100) if total > 0 else 0
        return accuracy, {'correct': correct, 'total': total, 'examples': examples}
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from model response."""
        # Priority 1: \boxed{} format
        boxed_matches = list(re.finditer(r'\\boxed\{([^}]+)\}', response))
        if boxed_matches:
            return self._clean_number(boxed_matches[-1].group(1))
        
        # Priority 2: #### pattern
        hash_match = re.search(r'####\s*([^\n]+)', response)
        if hash_match:
            return self._clean_number(hash_match.group(1))
        
        # Priority 3: "answer is X" pattern
        answer_match = re.search(
            r'(?:answer|result|total)\s*(?:is|=|:)\s*\$?([0-9,.]+)',
            response, re.IGNORECASE
        )
        if answer_match:
            return self._clean_number(answer_match.group(1))
        
        return None
    
    def get_expected_answer(self, sample: Dict) -> str:
        """Extract answer from GSM8K format (#### number)."""
        answer_text = sample['answer']
        match = re.search(r'####\s*([^\n]+)', answer_text)
        if match:
            return self._clean_number(match.group(1))
        return answer_text.strip()
    
    def _clean_number(self, s: str) -> str:
        """Clean number string: remove $, commas, spaces."""
        s = s.strip()
        s = s.replace('$', '').replace(',', '').replace(' ', '')
        s = s.lstrip('+-')
        match = re.search(r'^([0-9]+(?:\.[0-9]+)?)', s)
        if match:
            return match.group(1)
        return s
