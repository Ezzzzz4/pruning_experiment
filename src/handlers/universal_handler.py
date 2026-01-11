"""
Universal Handler - Dynamic Model Structure Discovery

Automatically discovers and analyzes the layer structure of ANY neural network.
Works with transformers, CNNs, encoder-decoder, multi-modal, and audio models.

For unusual architectures (YOLO, etc.), use specialized handlers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class ComponentInfo:
    """Information about a discovered model component."""
    path: str
    layers: nn.ModuleList
    count: int
    layer_type: str
    component_type: str  # 'encoder', 'decoder', 'vision', 'text', 'audio', 'main'


# Patterns for classifying components
COMPONENT_PATTERNS = {
    'encoder': ['encoder', 'encode', 'enc_', 'enc.'],
    'decoder': ['decoder', 'decode', 'dec_', 'dec.'],
    'vision': ['visual', 'vision', 'image', 'vit', 'patch_embed', 'img_'],
    'text': ['text', 'language', 'lm_', 'token', 'word'],
    'audio': ['audio', 'speech', 'mel', 'wav', 'acoustic'],
    'backbone': ['backbone', 'stem', 'features', 'body'],
    'head': ['head', 'classifier', 'fc', 'output'],
}

# Known layer group attribute names (for faster detection)
KNOWN_LAYER_PATTERNS = [
    'h', 'layers', 'layer', 'blocks', 'block', 'resblocks',
    'encoder_layers', 'decoder_layers', 'attn_layers',
]


class UniversalHandler:
    """
    Universal handler that auto-discovers model structure.
    
    Works with:
    - Standard transformers (GPT, BERT, LLaMA, Qwen, etc.)
    - Encoder-decoder models (DETR, Whisper, T5, etc.)
    - Multi-modal models (CLIP, SigLIP, LLaVA, etc.)
    - Vision transformers (ViT, DeiT, etc.)
    - Audio models (Wav2Vec2, HuBERT, etc.)
    
    For unusual architectures, use specialized handlers:
    - ResNetHandler: For ResNet's layer1/2/3/4 structure
    - YOLOHandler: For YOLO's backbone/neck/head structure
    
    Example:
        >>> handler = UniversalHandler(model)
        >>> handler.discover()
        Discovered 2 components:
          encoder: 6 layers at 'encoder.layers'
          decoder: 6 layers at 'decoder.layers'
        
        >>> results = handler.analyze(dataloader)
        >>> print(results['encoder']['bi_scores'])
    """
    
    def __init__(self, model: nn.Module, verbose: bool = True):
        """
        Initialize the universal handler.
        
        Args:
            model: PyTorch model to analyze
            verbose: Whether to print discovery info
        """
        self.model = model
        self.verbose = verbose
        self.device = next(model.parameters()).device
        
        # Discovered components
        self.components: Dict[str, ComponentInfo] = {}
        self.is_multimodal = False
        
        # Auto-discover on init
        self.discover()
    
    def discover(self) -> Dict[str, ComponentInfo]:
        """
        Discover all prunable components in the model.
        
        Returns:
            Dictionary of component_name -> ComponentInfo
        """
        # Find all layer groups
        layer_groups = self._find_layer_groups(self.model, "")
        
        if not layer_groups:
            raise ValueError(
                "Could not find any layer groups in model. "
                "Use a specialized handler for this architecture."
            )
        
        # Classify and store components
        self.components = {}
        for group in layer_groups:
            comp_type = self._classify_component(group['path'])
            comp_name = self._get_unique_name(comp_type)
            
            self.components[comp_name] = ComponentInfo(
                path=group['path'],
                layers=group['module'],
                count=group['count'],
                layer_type=group['layer_type'],
                component_type=comp_type
            )
        
        # Detect multi-modal
        self.is_multimodal = self._detect_multimodal()
        
        if self.verbose:
            self._print_discovery()
        
        return self.components
    
    def _find_layer_groups(
        self, 
        module: nn.Module, 
        path: str
    ) -> List[Dict]:
        """
        Recursively find all layer groups (ModuleList/Sequential with 2+ children).
        """
        groups = []
        
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            
            # Check if this is a layer group
            if isinstance(child, (nn.ModuleList, nn.Sequential)):
                if len(child) >= 2:
                    # Get type of first child as layer type
                    layer_type = type(list(child.children())[0]).__name__
                    
                    groups.append({
                        'path': child_path,
                        'module': child,
                        'count': len(child),
                        'layer_type': layer_type
                    })
            
            # Also check for known pattern names that might be ModuleLists
            if name in KNOWN_LAYER_PATTERNS:
                if hasattr(child, '__len__') and len(child) >= 2:
                    if isinstance(child, nn.Module) and child_path not in [g['path'] for g in groups]:
                        layer_type = type(list(child.children())[0]).__name__ if hasattr(child, 'children') else 'Unknown'
                        groups.append({
                            'path': child_path,
                            'module': child,
                            'count': len(child),
                            'layer_type': layer_type
                        })
            
            # Recurse into children (but not into already-found groups)
            if child_path not in [g['path'] for g in groups]:
                groups.extend(self._find_layer_groups(child, child_path))
        
        return groups
    
    def _classify_component(self, path: str) -> str:
        """
        Classify a component based on its path.
        
        Priority order:
        1. Modality-specific (vision, text, audio) - most specific
        2. Architectural (encoder, decoder) - generic
        3. Position-based (backbone, head) - fallback
        """
        path_lower = path.lower()
        
        # Priority 1: Check modality patterns first (most specific)
        modality_patterns = {
            'vision': ['visual', 'vision', 'image', 'vit', 'patch_embed', 'img_', 'vision_model'],
            'text': ['text', 'text_model', 'language', 'lm_', 'word'],
            'audio': ['audio', 'speech', 'mel', 'wav', 'acoustic'],
        }
        
        for comp_type, patterns in modality_patterns.items():
            if any(p in path_lower for p in patterns):
                return comp_type
        
        # Priority 2: Architectural patterns
        arch_patterns = {
            'encoder': ['encoder', 'encode', 'enc_', 'enc.'],
            'decoder': ['decoder', 'decode', 'dec_', 'dec.'],
        }
        
        for comp_type, patterns in arch_patterns.items():
            if any(p in path_lower for p in patterns):
                return comp_type
        
        # Priority 3: Position-based patterns
        position_patterns = {
            'backbone': ['backbone', 'stem', 'features', 'body'],
            'head': ['head', 'classifier', 'fc', 'output'],
        }
        
        for comp_type, patterns in position_patterns.items():
            if any(p in path_lower for p in patterns):
                return comp_type
        
        return 'main'
    
    def _get_unique_name(self, base_type: str) -> str:
        """Get unique component name (e.g., 'encoder', 'encoder_2')."""
        if base_type not in self.components:
            return base_type
        
        i = 2
        while f"{base_type}_{i}" in self.components:
            i += 1
        return f"{base_type}_{i}"
    
    def _detect_multimodal(self) -> bool:
        """Check if model has multiple modalities."""
        modal_types = {'vision', 'text', 'audio'}
        found_modals = set()
        
        for comp in self.components.values():
            if comp.component_type in modal_types:
                found_modals.add(comp.component_type)
        
        return len(found_modals) >= 2
    
    def _print_discovery(self) -> None:
        """Print discovery results."""
        print(f"\n✓ Discovered {len(self.components)} component(s):")
        for name, info in self.components.items():
            print(f"  {name}: {info.count} layers at '{info.path}' ({info.layer_type})")
        
        if self.is_multimodal:
            print("  ⭐ Multi-modal model detected!")
    
    def get_layers(self, component: str = 'main') -> nn.ModuleList:
        """Get layers for a specific component."""
        if component not in self.components:
            available = list(self.components.keys())
            raise ValueError(f"Component '{component}' not found. Available: {available}")
        return self.components[component].layers
    
    def get_layer_path(self, component: str = 'main') -> str:
        """Get path for a specific component."""
        if component not in self.components:
            available = list(self.components.keys())
            raise ValueError(f"Component '{component}' not found. Available: {available}")
        return self.components[component].path
    
    def list_components(self) -> List[str]:
        """List all discovered components."""
        return list(self.components.keys())
    
    def component_info(self, component: str) -> ComponentInfo:
        """Get info about a specific component."""
        return self.components[component]
    
    def compute_bi_scores(
        self,
        dataloader: torch.utils.data.DataLoader,
        component: str = 'main',
        num_samples: Optional[int] = 100
    ) -> Dict[int, float]:
        """
        Compute Block Influence scores for a component.
        
        Args:
            dataloader: DataLoader for samples
            component: Component to analyze ('main', 'encoder', 'vision', etc.)
            num_samples: Number of samples to use
            
        Returns:
            Dict mapping layer_idx -> BI score
        """
        from src.core.block_influence import BlockInfluenceAnalyzer
        
        analyzer = BlockInfluenceAnalyzer(self.model, verbose=False)
        layers = self.get_layers(component)
        path = self.get_layer_path(component)
        
        return analyzer.compute_bi_scores(dataloader, layers, path, num_samples)
    
    def analyze_all(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = 100
    ) -> Dict[str, Dict]:
        """
        Analyze all discovered components.
        
        Returns:
            Dict with BI scores and summary for each component
        """
        results = {}
        
        for comp_name in self.components:
            bi_scores = self.compute_bi_scores(dataloader, comp_name, num_samples)
            
            # Compute summary stats
            scores = list(bi_scores.values())
            redundant = [idx for idx, s in bi_scores.items() if s < 0.1]
            
            results[comp_name] = {
                'bi_scores': bi_scores,
                'redundant_layers': redundant,
                'summary': {
                    'total_layers': len(bi_scores),
                    'redundant_count': len(redundant),
                    'mean_bi': np.mean(scores),
                    'min_bi': np.min(scores),
                    'max_bi': np.max(scores),
                }
            }
        
        # Add cross-modal analysis for multimodal models
        if self.is_multimodal:
            results['cross_modal'] = self._cross_modal_analysis(results)
        
        return results
    
    def _cross_modal_analysis(self, results: Dict) -> Dict:
        """Compute cross-modal statistics for multimodal models."""
        modal_components = {}
        for name, info in self.components.items():
            if info.component_type in ['vision', 'text', 'audio']:
                modal_components[info.component_type] = name
        
        analysis = {'modalities': list(modal_components.keys())}
        
        # Compare redundancy between modalities
        if 'vision' in modal_components and 'text' in modal_components:
            vision_mean = results[modal_components['vision']]['summary']['mean_bi']
            text_mean = results[modal_components['text']]['summary']['mean_bi']
            analysis['vision_vs_text'] = {
                'vision_mean_bi': vision_mean,
                'text_mean_bi': text_mean,
                'more_redundant': 'vision' if vision_mean < text_mean else 'text',
                'difference': abs(vision_mean - text_mean)
            }
        
        return analysis
    
    def remove_layer(
        self,
        component: str,
        layer_idx: int,
        inplace: bool = False
    ) -> nn.Module:
        """
        Remove a layer from a specific component.
        
        Args:
            component: Component name
            layer_idx: Index of layer to remove
            inplace: Whether to modify model in-place
            
        Returns:
            Model with layer removed
        """
        import copy
        
        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)
        
        comp_info = self.components[component]
        path = comp_info.path
        
        # Navigate to parent and replace layers
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        current_layers = getattr(parent, parts[-1])
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(current_layers)
            if i != layer_idx
        ])
        
        setattr(parent, parts[-1], new_layers)
        
        if self.verbose:
            print(f"✓ Removed layer {layer_idx} from {component}")
            print(f"  {component} now has {len(new_layers)} layers")
        
        return model
    
    def remove_layers(
        self,
        component: str,
        layer_indices: List[int],
        inplace: bool = False
    ) -> nn.Module:
        """Remove multiple layers from a component."""
        import copy
        
        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)
        
        comp_info = self.components[component]
        path = comp_info.path
        
        # Navigate to parent
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        current_layers = getattr(parent, parts[-1])
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(current_layers)
            if i not in layer_indices
        ])
        
        setattr(parent, parts[-1], new_layers)
        
        if self.verbose:
            print(f"✓ Removed {len(layer_indices)} layers from {component}: {sorted(layer_indices)}")
            print(f"  {component} now has {len(new_layers)} layers")
        
        return model


def create_handler(model: nn.Module, **kwargs):
    """
    Factory function to create appropriate handler for a model.
    
    Returns UniversalHandler for most models, or specialized handler
    for unusual architectures.
    """
    # Check for ResNet-style
    if all(hasattr(model, f'layer{i}') for i in [1, 2, 3, 4]):
        from src.handlers.resnet_handler import ResNetLayerPruner
        if kwargs.get('verbose', True):
            print("Detected ResNet-style architecture, using ResNetLayerPruner")
        return ResNetLayerPruner(model, **kwargs)
    
    # Check for YOLO-style (if ultralytics model)
    if hasattr(model, 'model') and hasattr(model, 'predict'):
        try:
            from src.handlers.yolo_handler import YOLOHandler
            if kwargs.get('verbose', True):
                print("Detected YOLO-style architecture, using YOLOHandler")
            return YOLOHandler(model, **kwargs)
        except ImportError:
            pass
    
    # Default: Universal handler
    return UniversalHandler(model, **kwargs)
