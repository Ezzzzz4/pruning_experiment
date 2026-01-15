"""
Model Handlers for Universal Layer Pruning

Provides handlers for neural network layer manipulation:
- UniversalHandler: Works with most LLM architectures (auto-discovery)

Usage:
    from src.handlers import UniversalHandler
    
    handler = UniversalHandler(model)
    handler.remove_layers('main', [3, 4, 5])
"""

from src.handlers.universal_handler import UniversalHandler, create_handler

__all__ = [
    'UniversalHandler',
    'create_handler',
]
