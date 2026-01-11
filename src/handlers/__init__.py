"""
Model Handlers for Universal Layer Pruning

Provides handlers for different neural network architectures:
- UniversalHandler: Works with most architectures (auto-discovery)
- ResNetLayerPruner: For ResNet's layer1/2/3/4 structure
- YOLOHandler: For YOLO's backbone/neck/head structure

Usage:
    from src.handlers import create_handler
    
    handler = create_handler(model)  # Auto-selects appropriate handler
    results = handler.analyze_all(dataloader)
"""

from src.handlers.universal_handler import UniversalHandler, create_handler
from src.handlers.resnet_handler import ResNetLayerPruner
from src.handlers.yolo_handler import YOLOHandler

__all__ = [
    'UniversalHandler',
    'create_handler',
    'ResNetLayerPruner', 
    'YOLOHandler',
]
