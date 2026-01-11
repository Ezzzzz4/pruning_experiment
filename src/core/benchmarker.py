"""
Benchmarker - Quality and Speed Measurement

Provides task-specific metrics for evaluating pruned models:
- Language models: Perplexity, tokens/second
- Vision models: Top-k accuracy, images/second
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    quality_score: float
    throughput: float
    latency_ms: float
    memory_mb: float


class Benchmarker:
    """
    Task-specific benchmarking for pruned models.
    
    Measures quality metrics (perplexity, accuracy) and speed metrics
    (throughput, latency) with statistical rigor.
    
    Example:
        >>> benchmarker = Benchmarker(model, task_type='language')
        >>> results = benchmarker.full_benchmark(dataloader, num_runs=10)
        >>> print(results)
        {
            'quality': {'mean': 23.5, 'std': 0.3, 'ci_95': [23.2, 23.8]},
            'speed': {'mean': 1250.0, 'std': 50.0, 'ci_95': [1200, 1300]},
            ...
        }
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_type: str,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Initialize the benchmarker.
        
        Args:
            model: PyTorch model to benchmark
            task_type: 'language' or 'vision'
            device: Device to use ('cuda', 'cpu', or 'auto')
            verbose: Whether to print progress
        """
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.task_type = task_type
        self.verbose = verbose
        
    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    # ==================== QUALITY METRICS ====================
    
    def measure_perplexity(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Measure perplexity for language models.
        
        Perplexity = exp(average cross-entropy loss)
        Lower is better.
        
        Args:
            dataloader: DataLoader with tokenized text
            num_samples: Max batches to process
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = input_ids.clone()
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = None
                    labels = input_ids.clone()
                
                try:
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels if hasattr(self.model, 'lm_head') else None
                    )
                    
                    # Get loss
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        # Manual loss computation
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction='mean'
                        )
                    
                    total_loss += loss.item() * input_ids.numel()
                    total_tokens += input_ids.numel()
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error in perplexity computation: {e}")
                    continue
                
                samples_processed += 1
                if num_samples and samples_processed >= num_samples:
                    break
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def measure_accuracy(
        self,
        dataloader: torch.utils.data.DataLoader,
        top_k: int = 1,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Measure top-k accuracy for vision models.
        
        Args:
            dataloader: DataLoader with (images, labels)
            top_k: k for top-k accuracy (1 or 5)
            num_samples: Max batches to process
            
        Returns:
            Accuracy as fraction (0-1)
        """
        correct = 0
        total = 0
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    raise ValueError("Expected (images, labels) tuple for vision task")
                
                try:
                    outputs = self.model(images)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    if top_k == 1:
                        preds = logits.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                    else:
                        _, top_k_preds = logits.topk(top_k, dim=1)
                        correct += (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                    
                    total += labels.size(0)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error in accuracy computation: {e}")
                    continue
                
                samples_processed += 1
                if num_samples and samples_processed >= num_samples:
                    break
        
        return correct / total if total > 0 else 0.0
    
    def measure_quality(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Measure quality using task-appropriate metric.
        
        For language: perplexity (lower is better)
        For vision: accuracy (higher is better)
        """
        if self.task_type == 'language':
            return self.measure_perplexity(dataloader, num_samples)
        elif self.task_type == 'vision':
            return self.measure_accuracy(dataloader, top_k=1, num_samples=num_samples)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    # ==================== SPEED METRICS ====================
    
    def measure_throughput(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 50,
        warmup_batches: int = 5
    ) -> float:
        """
        Measure throughput (items per second).
        
        Args:
            dataloader: DataLoader for benchmarking
            num_batches: Number of batches to time
            warmup_batches: Warmup batches (not counted)
            
        Returns:
            Throughput in items/second
        """
        total_items = 0
        total_time = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare input
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items()
                             if isinstance(v, torch.Tensor)}
                    batch_size = inputs['input_ids'].size(0)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    batch_size = inputs.size(0)
                else:
                    inputs = batch.to(self.device)
                    batch_size = inputs.size(0)
                
                # Warmup
                if batch_count < warmup_batches:
                    try:
                        if isinstance(inputs, dict):
                            self.model(**inputs)
                        else:
                            self.model(inputs)
                    except:
                        pass
                    batch_count += 1
                    continue
                
                # Timed run
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                try:
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    else:
                        self.model(inputs)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Forward pass error: {e}")
                    continue
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                total_time += elapsed
                total_items += batch_size
                batch_count += 1
                
                if batch_count >= warmup_batches + num_batches:
                    break
        
        throughput = total_items / total_time if total_time > 0 else 0.0
        return throughput
    
    def measure_latency(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_runs: int = 100
    ) -> float:
        """
        Measure average latency per forward pass (milliseconds).
        """
        latencies = []
        run_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items()
                             if isinstance(v, torch.Tensor)}
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                try:
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    else:
                        self.model(inputs)
                except:
                    continue
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                latencies.append((time.perf_counter() - start) * 1000)  # ms
                run_count += 1
                
                if run_count >= num_runs:
                    break
        
        return np.mean(latencies) if latencies else 0.0
    
    def measure_memory(self) -> float:
        """Measure peak GPU memory usage in MB."""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0
    
    # ==================== COMBINED BENCHMARK ====================
    
    def full_benchmark(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_runs: int = 10,
        num_samples: Optional[int] = 100
    ) -> Dict:
        """
        Run complete benchmark with statistical analysis.
        
        Args:
            dataloader: DataLoader for benchmarking
            num_runs: Number of runs for statistical significance
            num_samples: Samples per run for quality measurement
            
        Returns:
            Dictionary with quality, speed, and memory metrics
        """
        quality_scores = []
        throughputs = []
        latencies = []
        
        iterator = range(num_runs)
        if self.verbose:
            iterator = tqdm(iterator, desc="Benchmarking")
        
        for _ in iterator:
            # Quality
            quality = self.measure_quality(dataloader, num_samples)
            quality_scores.append(quality)
            
            # Speed
            throughput = self.measure_throughput(dataloader, num_batches=20)
            throughputs.append(throughput)
            
            # Latency
            latency = self.measure_latency(dataloader, num_runs=20)
            latencies.append(latency)
        
        # Memory
        memory_mb = self.measure_memory()
        
        # Parameter count
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Compute statistics
        def compute_stats(values: List[float]) -> Dict:
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr)
            ci_95 = 1.96 * std / np.sqrt(len(arr))
            return {
                'mean': float(mean),
                'std': float(std),
                'ci_95': [float(mean - ci_95), float(mean + ci_95)],
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
        
        results = {
            'quality': compute_stats(quality_scores),
            'speed': compute_stats(throughputs),
            'latency_ms': compute_stats(latencies),
            'memory_mb': memory_mb,
            'num_params': num_params,
            'task_type': self.task_type,
            'num_runs': num_runs,
        }
        
        if self.verbose:
            print(f"\n📊 Benchmark Results:")
            print(f"   Quality: {results['quality']['mean']:.4f} ± {results['quality']['std']:.4f}")
            print(f"   Throughput: {results['speed']['mean']:.1f} ± {results['speed']['std']:.1f} items/sec")
            print(f"   Latency: {results['latency_ms']['mean']:.2f} ± {results['latency_ms']['std']:.2f} ms")
            print(f"   Memory: {results['memory_mb']:.1f} MB")
            print(f"   Parameters: {results['num_params']:,}")
        
        return results


def compare_models(
    original_model: nn.Module,
    pruned_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task_type: str,
    num_runs: int = 10
) -> Dict:
    """
    Compare original and pruned models side-by-side.
    
    Returns dict with metrics for both and percentage changes.
    """
    original_bench = Benchmarker(original_model, task_type, verbose=False)
    pruned_bench = Benchmarker(pruned_model, task_type, verbose=False)
    
    print("Benchmarking original model...")
    original_results = original_bench.full_benchmark(dataloader, num_runs)
    
    print("Benchmarking pruned model...")
    pruned_results = pruned_bench.full_benchmark(dataloader, num_runs)
    
    # Compute deltas
    quality_delta = (
        (pruned_results['quality']['mean'] - original_results['quality']['mean']) /
        original_results['quality']['mean']
    )
    speed_delta = (
        (pruned_results['speed']['mean'] - original_results['speed']['mean']) /
        original_results['speed']['mean']
    )
    param_delta = (
        (pruned_results['num_params'] - original_results['num_params']) /
        original_results['num_params']
    )
    
    comparison = {
        'original': original_results,
        'pruned': pruned_results,
        'delta': {
            'quality': quality_delta,
            'speed': speed_delta,
            'params': param_delta,
        }
    }
    
    print(f"\n📈 Comparison:")
    print(f"   Quality change: {quality_delta:+.2%}")
    print(f"   Speed change: {speed_delta:+.2%}")
    print(f"   Parameter reduction: {-param_delta:.2%}")
    
    return comparison
