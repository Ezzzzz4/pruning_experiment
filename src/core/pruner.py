"""
Universal Layer Pruner

Automatically detects and prunes layers from various neural network architectures.
Uses pattern matching to identify layer structures in different model types.

Supported architectures:
- GPT-style (transformer.h)
- BERT/ViT-style (encoder.layer)
- LLaMA-style (model.layers)
- Block-based (blocks)
- Sequential (features)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import copy
from dataclasses import dataclass

from src.core.block_influence import BlockInfluenceAnalyzer


@dataclass
class LayerAnalysis:
    """Results from analyzing a single layer."""
    layer_idx: int
    bi_score: float
    removal_impact: Optional[float] = None
    is_redundant: bool = False
    rank: int = 0


# Pattern: (attribute_path, description, model_examples)
DETECTION_PATTERNS = [
    ('h', 'GPT-style (direct)', ['gpt2']),  # GPT-2 from transformers
    ('transformer.h', 'GPT-style (nested)', ['gpt-neo', 'bloom']),
    ('encoder.layer', 'BERT/ViT-style', ['bert', 'vit', 'roberta', 'electra']),
    ('encoder.layers', 'Encoder-layers', ['whisper']),
    ('model.layers', 'LLaMA-style', ['llama', 'mistral', 'qwen']),
    ('decoder.layers', 'Decoder-only', ['opt', 'marian']),
    ('layers', 'Simple layers', ['generic']),
    ('blocks', 'Block-based', ['efficientnet', 'convnext']),
    ('features', 'Sequential', ['vgg', 'mobilenet']),
]


class UniversalLayerPruner:
    """
    Universal layer pruner with automatic architecture detection.
    
    Works with most transformer and vision architectures by detecting
    common layer organization patterns. For ResNet-style models with
    layer1/2/3/4 structure, use ResNetLayerPruner instead.
    
    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained('gpt2')
        >>> pruner = UniversalLayerPruner(model, task_type='language')
        # ✓ Detected GPT-style architecture
        # Found 12 layers at 'transformer.h'
        
        >>> importance = pruner.analyze_layer_importance(dataset)
        >>> pruned = pruner.remove_layers([3, 4, 5])  # Remove redundant layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_type: str = 'auto',
        verbose: bool = True
    ):
        """
        Initialize the universal layer pruner.
        
        Args:
            model: PyTorch model to prune
            task_type: 'language', 'vision', or 'auto' (auto-detect)
            verbose: Whether to print detection and pruning info
        """
        self.original_model = model
        self.model = model
        self.verbose = verbose
        self.device = next(model.parameters()).device
        
        # Detect layer structure
        self.layers, self.layer_path = self._auto_detect_layers()
        self.n_layers = len(self.layers)
        
        # Detect/set task type
        self.task_type = self._resolve_task_type(task_type)
        
        # Initialize analyzer
        self.bi_analyzer = BlockInfluenceAnalyzer(model, verbose=False)
        
        if self.verbose:
            print(f"✓ UniversalLayerPruner initialized")
            print(f"  Model has {self.n_layers} layers at '{self.layer_path}'")
            print(f"  Task type: {self.task_type}")
    
    def _get_nested_attr(self, obj: object, attr_path: str) -> object:
        """
        Get nested attribute using dot notation.
        
        Example: _get_nested_attr(model, 'transformer.h') -> model.transformer.h
        """
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    def _set_nested_attr(self, obj: object, attr_path: str, value: object) -> None:
        """Set nested attribute using dot notation."""
        parts = attr_path.split('.')
        for attr in parts[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, parts[-1], value)
    
    def _auto_detect_layers(self) -> Tuple[nn.ModuleList, str]:
        """
        Automatically detect the layer structure of the model.
        
        Returns:
            Tuple of (layers ModuleList, path string)
            
        Raises:
            ValueError: If layers cannot be detected
        """
        for pattern, description, examples in DETECTION_PATTERNS:
            try:
                layers = self._get_nested_attr(self.model, pattern)
                if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                    if self.verbose:
                        print(f"✓ Detected {description} architecture")
                        print(f"  Found {len(layers)} layers at '{pattern}'")
                    # Convert Sequential to ModuleList if needed
                    if isinstance(layers, nn.Sequential):
                        layers = nn.ModuleList(list(layers.children()))
                    return layers, pattern
            except AttributeError:
                continue
        
        # Check for ResNet-style
        if all(hasattr(self.model, f'layer{i}') for i in [1, 2, 3, 4]):
            raise ValueError(
                "Detected ResNet-style architecture (layer1/2/3/4). "
                "Please use ResNetLayerPruner instead."
            )
        
        raise ValueError(
            f"Could not auto-detect layer structure.\n"
            f"Supported patterns: {[p[0] for p in DETECTION_PATTERNS]}\n"
            f"For ResNet, use ResNetLayerPruner."
        )
    
    def _resolve_task_type(self, task_type: str) -> str:
        """Resolve 'auto' task type based on model architecture."""
        if task_type != 'auto':
            return task_type
        
        # Infer from layer path
        if 'encoder.layer' in self.layer_path:
            # Could be BERT (language) or ViT (vision)
            # Check for vision-specific attributes
            if hasattr(self.model, 'patch_embed') or hasattr(self.model, 'embeddings'):
                if hasattr(self.model.embeddings, 'patch_embeddings'):
                    return 'vision'
            return 'language'
        elif 'transformer' in self.layer_path or 'decoder' in self.layer_path:
            return 'language'
        elif 'blocks' in self.layer_path or 'features' in self.layer_path:
            return 'vision'
        
        return 'language'  # Default
    
    def get_layer_info(self) -> Dict[str, any]:
        """Get information about the model's layer structure."""
        return {
            'n_layers': self.n_layers,
            'layer_path': self.layer_path,
            'task_type': self.task_type,
            'layer_types': [type(layer).__name__ for layer in self.layers],
        }
    
    def analyze_layer_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: str = 'block_influence',
        num_samples: Optional[int] = 100,
        **kwargs
    ) -> Dict[int, LayerAnalysis]:
        """
        Analyze the importance of each layer.
        
        Args:
            dataloader: DataLoader providing input samples
            method: 'block_influence' (fast) or 'removal_impact' (accurate but slow)
            num_samples: Number of samples to use for analysis
            
        Returns:
            Dictionary mapping layer_idx to LayerAnalysis
        """
        if self.verbose:
            print(f"\nAnalyzing layer importance using '{method}' method...")
        
        if method == 'block_influence':
            bi_scores = self.bi_analyzer.compute_bi_scores(
                dataloader, self.layers, self.layer_path, num_samples
            )
            
            # Rank layers
            ranked = sorted(bi_scores.items(), key=lambda x: x[1], reverse=True)
            rank_map = {idx: rank for rank, (idx, _) in enumerate(ranked)}
            
            # Identify redundant layers
            redundant = self.bi_analyzer.identify_redundant_layers(bi_scores)
            
            results = {}
            for idx, bi_score in bi_scores.items():
                results[idx] = LayerAnalysis(
                    layer_idx=idx,
                    bi_score=bi_score,
                    is_redundant=(idx in redundant),
                    rank=rank_map[idx]
                )
            
            if self.verbose:
                print(f"\n📊 Layer Importance Summary:")
                print(f"   Most important: Layer {ranked[0][0]} (BI={ranked[0][1]:.3f})")
                print(f"   Least important: Layer {ranked[-1][0]} (BI={ranked[-1][1]:.3f})")
                print(f"   Identified {len(redundant)} redundant layers: {redundant}")
            
            return results
            
        elif method == 'removal_impact':
            # TODO: Implement actual removal and benchmark for each layer
            raise NotImplementedError("removal_impact method not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def remove_layers(
        self,
        indices: List[int],
        inplace: bool = False
    ) -> nn.Module:
        """
        Remove layers at specified indices.
        
        Args:
            indices: List of layer indices to remove
            inplace: If True, modify model in-place; otherwise return copy
            
        Returns:
            Model with specified layers removed
        """
        if not indices:
            return self.model if inplace else copy.deepcopy(self.model)
        
        # Validate indices
        invalid = [i for i in indices if i < 0 or i >= self.n_layers]
        if invalid:
            raise ValueError(f"Invalid layer indices: {invalid}. Valid range: 0-{self.n_layers-1}")
        
        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)
        
        # Get current layers
        current_layers = self._get_nested_attr(model, self.layer_path)
        
        # Create new layer list excluding removed indices
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(current_layers)
            if i not in indices
        ])
        
        # Set new layers
        self._set_nested_attr(model, self.layer_path, new_layers)
        
        if self.verbose:
            print(f"✓ Removed {len(indices)} layers: {sorted(indices)}")
            print(f"  Model now has {len(new_layers)} layers")
        
        # Update internal state if inplace
        if inplace:
            self.layers = new_layers
            self.n_layers = len(new_layers)
        
        return model
    
    def restore_model(self) -> nn.Module:
        """Restore the original unpruned model."""
        self.model = copy.deepcopy(self.original_model)
        self.layers, self.layer_path = self._auto_detect_layers()
        self.n_layers = len(self.layers)
        if self.verbose:
            print("✓ Model restored to original state")
        return self.model
    
    def get_redundant_layers(
        self,
        analysis: Dict[int, LayerAnalysis]
    ) -> List[int]:
        """Get list of redundant layer indices from analysis results."""
        return [idx for idx, result in analysis.items() if result.is_redundant]
    
    def auto_optimize(
        self,
        dataloader: torch.utils.data.DataLoader,
        benchmarker,  # Benchmarker instance
        target_speedup: float = 0.3,
        max_quality_loss: float = 0.05,
        num_samples: int = 100
    ) -> Tuple[nn.Module, List[int], Dict]:
        """
        Automatically find optimal pruning configuration.
        
        Uses binary search to find the best trade-off between speed and quality.
        
        Args:
            dataloader: DataLoader for analysis and benchmarking
            benchmarker: Benchmarker instance for quality/speed measurement
            target_speedup: Target speedup ratio (e.g., 0.3 = 30% faster)
            max_quality_loss: Maximum acceptable quality degradation
            num_samples: Samples to use for analysis
            
        Returns:
            Tuple of (pruned_model, removed_layer_indices, metrics_dict)
        """
        if self.verbose:
            print(f"\n🔍 Auto-optimizing for {target_speedup*100:.0f}% speedup, <{max_quality_loss*100:.0f}% quality loss...")
        
        # Analyze layer importance
        analysis = self.analyze_layer_importance(dataloader, num_samples=num_samples)
        
        # Get baseline metrics
        baseline = benchmarker.full_benchmark(dataloader)
        baseline_quality = baseline['quality']['mean']
        baseline_speed = baseline['speed']['mean']
        
        if self.verbose:
            print(f"\n📏 Baseline: Quality={baseline_quality:.4f}, Speed={baseline_speed:.2f}")
        
        # Sort layers by importance (ascending = most redundant first)
        sorted_layers = sorted(
            analysis.items(),
            key=lambda x: x[1].bi_score
        )
        
        # Binary search for optimal number of layers to remove
        best_config = {
            'removed': [],
            'model': self.model,
            'speedup': 0,
            'quality_loss': 0,
            'metrics': baseline
        }
        
        # Try removing increasingly more layers
        for n_remove in range(1, len(sorted_layers) // 2):
            # Get n most redundant layers (excluding first 2 and last layer)
            candidates = [
                idx for idx, _ in sorted_layers
                if idx >= 2 and idx < self.n_layers - 1
            ][:n_remove]
            
            if not candidates:
                continue
            
            # Create pruned model and benchmark
            pruned_model = self.remove_layers(candidates, inplace=False)
            
            try:
                # Update benchmarker model
                benchmarker.model = pruned_model
                metrics = benchmarker.full_benchmark(dataloader)
                
                quality = metrics['quality']['mean']
                speed = metrics['speed']['mean']
                
                quality_loss = (quality - baseline_quality) / baseline_quality
                speedup = (speed - baseline_speed) / baseline_speed
                
                if self.verbose:
                    print(f"  Trying {n_remove} layers: Δquality={quality_loss:+.2%}, Δspeed={speedup:+.2%}")
                
                # Check if this meets constraints and is better
                if quality_loss <= max_quality_loss and speedup >= best_config['speedup']:
                    best_config = {
                        'removed': candidates,
                        'model': pruned_model,
                        'speedup': speedup,
                        'quality_loss': quality_loss,
                        'metrics': metrics
                    }
                    
                    if speedup >= target_speedup:
                        break  # Found good enough solution
                        
            except Exception as e:
                if self.verbose:
                    print(f"  Error with {n_remove} layers: {e}")
                continue
        
        if self.verbose:
            print(f"\n✓ Optimal config: Remove {len(best_config['removed'])} layers")
            print(f"  Layers: {best_config['removed']}")
            print(f"  Speedup: {best_config['speedup']:+.1%}")
            print(f"  Quality loss: {best_config['quality_loss']:+.2%}")
        
        return best_config['model'], best_config['removed'], best_config['metrics']


def create_pruner(model: nn.Module, **kwargs):
    """
    Factory function to create the appropriate pruner/handler for a model.
    
    Returns:
    - ResNetLayerPruner for ResNet-style (layer1/2/3/4)
    - YOLOHandler for YOLO-style
    - UniversalHandler for encoder-decoder, multi-modal
    - UniversalLayerPruner for simple transformers
    
    For maximum flexibility, use UniversalHandler from src.handlers.
    """
    # Check for ResNet-style
    if all(hasattr(model, f'layer{i}') for i in [1, 2, 3, 4]):
        from src.handlers.resnet_handler import ResNetLayerPruner
        if kwargs.get('verbose', True):
            print("Detected ResNet-style architecture, using ResNetLayerPruner")
        return ResNetLayerPruner(model, **kwargs)
    
    # Check for YOLO-style
    if hasattr(model, 'model') and hasattr(model, 'predict'):
        try:
            from src.handlers.yolo_handler import YOLOHandler
            if kwargs.get('verbose', True):
                print("Detected YOLO-style architecture, using YOLOHandler")
            return YOLOHandler(model, **kwargs)
        except ImportError:
            pass
    
    # Try UniversalLayerPruner first (for simple transformers)
    try:
        return UniversalLayerPruner(model, **kwargs)
    except ValueError:
        # Fall back to UniversalHandler (for complex multi-component models)
        from src.handlers.universal_handler import UniversalHandler
        if kwargs.get('verbose', True):
            print("Using UniversalHandler for complex architecture")
        return UniversalHandler(model, **kwargs)

