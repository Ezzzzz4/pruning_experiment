"""
Benchmark plugin registry and auto-discovery.

Available Benchmarks:
- gsm8k: Math reasoning benchmark (GSM8K dataset)
- language: Language model perplexity benchmark
- vision: Zero-shot image classification (CIFAR-10)

Usage:
    from experiments.benchmarks import get_benchmark, list_benchmarks
    
    # Get a benchmark instance
    benchmark = get_benchmark('vision')
    
    # Load dataset and evaluate
    dataset = benchmark.load_dataset(num_samples=100)
    score, details = benchmark.evaluate(model, processor, dataset)

Creating Custom Benchmarks:
    1. Copy experiments/benchmarks/custom_template.py to your_benchmark.py
    2. Implement load_dataset(), evaluate(), extract_answer()
    3. Register below in BENCHMARKS dict
    4. Run: python benchmark.py --model <model> --benchmark your_benchmark
"""

from .base import BaseBenchmark
from .gsm8k import GSM8KBenchmark
from .language import LanguageBenchmark
from .vision import VisionBenchmark

# Registry of available benchmarks
# Add custom benchmarks here: 'name': BenchmarkClass
BENCHMARKS = {
    'gsm8k': GSM8KBenchmark,
    'language': LanguageBenchmark,
    'vision': VisionBenchmark,
    # Example: 'my_benchmark': MyBenchmark,
}


def get_benchmark(name: str) -> BaseBenchmark:
    """Get a benchmark instance by name."""
    if name not in BENCHMARKS:
        available = ', '.join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARKS[name]()


def list_benchmarks() -> list:
    """List all available benchmark names."""
    return list(BENCHMARKS.keys())

