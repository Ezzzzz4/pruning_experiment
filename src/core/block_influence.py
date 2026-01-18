"""
Block Influence (BI) Metric Implementation

Computes how much each layer transforms its input, used to identify redundant layers.

Reference: ShortGPT (Men et al., 2024) - "Layers in Large Language Models are More Redundant Than You Expect"

BI Formula:
    BI(layer) = 1 - cosine_similarity(input_hidden, output_hidden)
    
    - High BI (→1): Layer significantly transforms input = IMPORTANT
    - Low BI (→0): Layer barely changes input = REDUNDANT (safe to remove)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class LayerActivation:
    """Stores input and output activations for a layer."""
    input_hidden: torch.Tensor
    output_hidden: torch.Tensor


class BlockInfluenceAnalyzer:
    """
    Computes Block Influence (BI) scores for neural network layers.
    
    The BI metric measures how much a layer transforms its input:
    - High BI = layer is important (significantly transforms data)
    - Low BI = layer is redundant (minimal transformation, safe to prune)
    
    Example:
        >>> analyzer = BlockInfluenceAnalyzer(model)
        >>> bi_scores = analyzer.compute_bi_scores(dataloader)
        >>> print(bi_scores)
        {0: 0.35, 1: 0.12, 2: 0.48, 3: 0.05, ...}  # Layer 3 is redundant
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Initialize the Block Influence analyzer.
        
        Args:
            model: PyTorch model to analyze
            device: Device to use ('cuda', 'cpu', or 'auto')
            verbose: Whether to print progress information
        """
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.verbose = verbose
        
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[str, LayerActivation] = {}
        
    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _create_hook(self, layer_name: str) -> Callable:
        """Create a forward hook that captures layer input and output."""
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            # Handle different input formats
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
            else:
                input_tensor = input
                
            # Handle different output formats (some layers return tuples)
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            
            # Store activations (detached to save memory)
            self._activations[layer_name] = LayerActivation(
                input_hidden=input_tensor.detach(),
                output_hidden=output_tensor.detach()
            )
        return hook
    
    def register_hooks(self, layers: nn.ModuleList, layer_path: str) -> None:
        """
        Register forward hooks on all layers to capture activations.
        
        Args:
            layers: ModuleList of layers to analyze
            layer_path: Base path for naming (e.g., 'transformer.h')
        """
        self.clear_hooks()
        
        for idx, layer in enumerate(layers):
            layer_name = f"{layer_path}.{idx}"
            hook = layer.register_forward_hook(self._create_hook(layer_name))
            self._hooks.append(hook)
            
        if self.verbose:
            print(f"✓ Registered hooks on {len(layers)} layers")
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()
    
    def _compute_cosine_similarity(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two tensors.
        
        Handles arbitrary tensor shapes by flattening.
        """
        # Flatten to 2D: (batch, features)
        a_flat = tensor_a.reshape(tensor_a.size(0), -1)
        b_flat = tensor_b.reshape(tensor_b.size(0), -1)
        
        # Compute cosine similarity per sample, then average
        cos_sim = F.cosine_similarity(a_flat, b_flat, dim=1)
        return cos_sim.mean().item()
    
    def compute_bi_scores(
        self,
        dataloader: torch.utils.data.DataLoader,
        layers: nn.ModuleList,
        layer_path: str,
        num_samples: Optional[int] = None,
        aggregate: str = 'mean'
    ) -> Dict[int, float]:
        """
        Compute Block Influence scores for all layers.
        
        Args:
            dataloader: DataLoader providing input samples
            layers: ModuleList of layers to analyze
            layer_path: Path to layers (e.g., 'transformer.h')
            num_samples: Max samples to use (None = all)
            aggregate: How to aggregate across samples ('mean' or 'median')
            
        Returns:
            Dictionary mapping layer_idx -> BI score (0-1)
        """
        self.register_hooks(layers, layer_path)
        
        # Collect BI scores for each sample
        all_bi_scores: Dict[int, List[float]] = {i: [] for i in range(len(layers))}
        samples_processed = 0
        
        iterator = tqdm(dataloader, desc="Computing BI scores") if self.verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                # Handle different batch formats
                if isinstance(batch, dict):
                    # HuggingFace style
                    inputs = {k: v.to(self.device) for k, v in batch.items() 
                             if isinstance(v, torch.Tensor)}
                elif isinstance(batch, (list, tuple)):
                    # Standard PyTorch style
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass to capture activations
                try:
                    if isinstance(inputs, dict):
                        # Check if this is a vision-only input for a multi-modal model (CLIP, SigLIP)
                        has_pixel_values = 'pixel_values' in inputs
                        has_input_ids = 'input_ids' in inputs
                        
                        if has_pixel_values and not has_input_ids:
                            # Vision-only input - try sub-model methods
                            if hasattr(self.model, 'get_image_features'):
                                # CLIP/SigLIP style
                                self.model.get_image_features(**inputs)
                            elif hasattr(self.model, 'vision_model'):
                                # Direct vision model access
                                self.model.vision_model(**inputs)
                            else:
                                # Fallback
                                self.model(**inputs)
                        else:
                            self.model(**inputs)
                    else:
                        self.model(inputs)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Forward pass error: {e}")
                    continue
                
                # Compute BI for each layer from captured activations
                for idx in range(len(layers)):
                    layer_name = f"{layer_path}.{idx}"
                    if layer_name in self._activations:
                        activation = self._activations[layer_name]
                        cos_sim = self._compute_cosine_similarity(
                            activation.input_hidden,
                            activation.output_hidden
                        )
                        bi_score = 1.0 - cos_sim  # High similarity = low influence
                        all_bi_scores[idx].append(bi_score)
                
                samples_processed += 1
                if num_samples and samples_processed >= num_samples:
                    break
        
        self.clear_hooks()
        
        # Aggregate scores
        bi_scores: Dict[int, float] = {}
        for idx, scores in all_bi_scores.items():
            if scores:
                if aggregate == 'mean':
                    bi_scores[idx] = np.mean(scores)
                elif aggregate == 'median':
                    bi_scores[idx] = np.median(scores)
                else:
                    bi_scores[idx] = np.mean(scores)
            else:
                bi_scores[idx] = 0.0
        
        if self.verbose:
            print(f"✓ Computed BI scores for {len(bi_scores)} layers")
            print(f"  Range: [{min(bi_scores.values()):.3f}, {max(bi_scores.values()):.3f}]")
        
        return bi_scores
    
    def compute_bi_with_stats(
        self,
        dataloader: torch.utils.data.DataLoader,
        layers: nn.ModuleList,
        layer_path: str,
        num_samples: Optional[int] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute BI scores with statistical information.
        
        Returns dict with mean, std, min, max for each layer.
        """
        self.register_hooks(layers, layer_path)
        
        all_bi_scores: Dict[int, List[float]] = {i: [] for i in range(len(layers))}
        samples_processed = 0
        
        iterator = tqdm(dataloader, desc="Computing BI scores") if self.verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() 
                             if isinstance(v, torch.Tensor)}
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                try:
                    if isinstance(inputs, dict):
                        self.model(**inputs)
                    else:
                        self.model(inputs)
                except:
                    continue
                
                for idx in range(len(layers)):
                    layer_name = f"{layer_path}.{idx}"
                    if layer_name in self._activations:
                        activation = self._activations[layer_name]
                        cos_sim = self._compute_cosine_similarity(
                            activation.input_hidden,
                            activation.output_hidden
                        )
                        all_bi_scores[idx].append(1.0 - cos_sim)
                
                samples_processed += 1
                if num_samples and samples_processed >= num_samples:
                    break
        
        self.clear_hooks()
        
        # Compute statistics
        result: Dict[int, Dict[str, float]] = {}
        for idx, scores in all_bi_scores.items():
            if scores:
                arr = np.array(scores)
                result[idx] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'n_samples': len(scores)
                }
            else:
                result[idx] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0,
                    'max': 0.0, 'median': 0.0, 'n_samples': 0
                }
        
        return result
    
    def rank_layers_by_importance(
        self,
        bi_scores: Dict[int, float]
    ) -> List[Tuple[int, float]]:
        """
        Rank layers by importance (highest BI first = most important).
        
        Args:
            bi_scores: Dictionary of layer_idx -> BI score
            
        Returns:
            List of (layer_idx, bi_score) sorted by importance (descending)
        """
        return sorted(bi_scores.items(), key=lambda x: x[1], reverse=True)
    
    def identify_redundant_layers(
        self,
        bi_scores: Dict[int, float],
        threshold: float = 0.1,
        max_ratio: float = 0.3
    ) -> List[int]:
        """
        Identify layers that are likely redundant (safe to remove).
        
        Args:
            bi_scores: Dictionary of layer_idx -> BI score
            threshold: BI score below which a layer is considered redundant
            max_ratio: Maximum ratio of layers to mark as redundant
            
        Returns:
            List of layer indices considered redundant
        """
        n_layers = len(bi_scores)
        max_redundant = int(n_layers * max_ratio)
        
        # Sort by BI score (ascending = most redundant first)
        sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
        
        redundant = []
        for idx, score in sorted_layers:
            if score < threshold and len(redundant) < max_redundant:
                # Never mark first 2 or last layer as redundant
                if idx >= 2 and idx < n_layers - 1:
                    redundant.append(idx)
        
        return redundant


def compute_block_influence(
    model: nn.Module,
    layer_idx: int,
    layers: nn.ModuleList,
    layer_path: str,
    sample_input: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Convenience function to compute BI for a single layer with one input.
    
    Args:
        model: The neural network model
        layer_idx: Index of layer to analyze
        layers: ModuleList containing the layers
        layer_path: Path to layers
        sample_input: Single input tensor
        device: Device to use
        
    Returns:
        Block Influence score (0-1)
    """
    analyzer = BlockInfluenceAnalyzer(model, device=device, verbose=False)
    
    # Create simple dataloader with single sample
    class SingleSampleDataset(torch.utils.data.Dataset):
        def __init__(self, tensor):
            self.tensor = tensor
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.tensor
    
    dataset = SingleSampleDataset(sample_input)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    bi_scores = analyzer.compute_bi_scores(dataloader, layers, layer_path, num_samples=1)
    
    return bi_scores.get(layer_idx, 0.0)
