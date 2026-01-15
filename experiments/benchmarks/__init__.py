"""
Benchmark plugin registry and auto-discovery.

Usage:
    from experiments.benchmarks import get_benchmark, list_benchmarks
    
    benchmark = get_benchmark('gsm8k')
    result = benchmark.evaluate(model, tokenizer, num_samples=50)
"""

from .base import BaseBenchmark
from .gsm8k import GSM8KBenchmark
from .language import LanguageBenchmark

# Registry of available benchmarks
BENCHMARKS = {
    'gsm8k': GSM8KBenchmark,
    'language': LanguageBenchmark,
}


def get_benchmark(name: str) -> BaseBenchmark:
    """Get a benchmark class by name."""
    if name not in BENCHMARKS:
        available = ', '.join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARKS[name]()


def list_benchmarks() -> list:
    """List all available benchmark names."""
    return list(BENCHMARKS.keys())
