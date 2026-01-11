"""
ResNet Layer Pruner

Specialized handler for ResNet-style architectures with grouped layer structure:
- layer1: 2-3 blocks (64 channels, stride 1)
- layer2: 3-4 blocks (128 channels, stride 2)
- layer3: 4-6 blocks (256 channels, stride 2)
- layer4: 2-3 blocks (512 channels, stride 2)

Each "layer" is a Sequential containing multiple BasicBlock or Bottleneck modules.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np
from dataclasses import dataclass


@dataclass
class BlockAnalysis:
    """Analysis results for a single ResNet block."""
    layer_group: str  # 'layer1', 'layer2', etc.
    block_idx: int
    bi_score: float
    is_redundant: bool = False


class ResNetLayerPruner:
    """
    Specialized pruner for ResNet-style architectures.
    
    ResNet has a unique grouped structure that requires special handling:
    - Cannot easily remove entire layer groups (breaks channel dimensions)
    - Should remove individual blocks within groups
    - layer1 and layer4 are typically more important
    - layer3 often has most redundancy (largest group)
    
    Example:
        >>> from torchvision.models import resnet50
        >>> model = resnet50(pretrained=True)
        >>> pruner = ResNetLayerPruner(model)
        # ✓ Detected ResNet-style architecture
        # layer1: 3 blocks, layer2: 4 blocks, layer3: 6 blocks, layer4: 3 blocks
        
        >>> importance = pruner.analyze_block_importance(dataloader)
        >>> pruned = pruner.remove_block('layer3', 2)  # Remove most redundant
    """
    
    LAYER_GROUPS = ['layer1', 'layer2', 'layer3', 'layer4']
    
    def __init__(
        self,
        model: nn.Module,
        verbose: bool = True
    ):
        """
        Initialize the ResNet pruner.
        
        Args:
            model: PyTorch ResNet model
            verbose: Whether to print info
            
        Raises:
            ValueError: If model is not ResNet-style
        """
        self.model = model
        self.verbose = verbose
        self.device = next(model.parameters()).device
        
        self._verify_resnet_structure()
        self.block_counts = self._count_blocks()
        
        if verbose:
            print("✓ ResNetLayerPruner initialized")
            for group, count in self.block_counts.items():
                print(f"  {group}: {count} blocks")
    
    def _verify_resnet_structure(self) -> None:
        """Verify the model has ResNet-style structure."""
        missing = [g for g in self.LAYER_GROUPS if not hasattr(self.model, g)]
        if missing:
            raise ValueError(
                f"Model missing layer groups: {missing}. "
                f"This doesn't appear to be a ResNet-style model."
            )
    
    def _count_blocks(self) -> Dict[str, int]:
        """Count blocks in each layer group."""
        return {
            group: len(list(getattr(self.model, group).children()))
            for group in self.LAYER_GROUPS
        }
    
    def get_block(self, layer_group: str, block_idx: int) -> nn.Module:
        """Get a specific block from a layer group."""
        layer = getattr(self.model, layer_group)
        blocks = list(layer.children())
        if block_idx >= len(blocks):
            raise ValueError(
                f"block_idx {block_idx} out of range for {layer_group} "
                f"(has {len(blocks)} blocks)"
            )
        return blocks[block_idx]
    
    def _compute_block_bi(
        self,
        block: nn.Module,
        sample_input: torch.Tensor
    ) -> float:
        """Compute Block Influence for a single block."""
        with torch.no_grad():
            input_hidden = sample_input.clone()
            output_hidden = block(sample_input)
            
            # Flatten and compute cosine similarity
            in_flat = input_hidden.view(input_hidden.size(0), -1)
            out_flat = output_hidden.view(output_hidden.size(0), -1)
            
            cos_sim = torch.nn.functional.cosine_similarity(in_flat, out_flat, dim=1)
            bi_score = 1.0 - cos_sim.mean().item()
            
        return bi_score
    
    def analyze_block_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict[str, Dict[int, BlockAnalysis]]:
        """
        Analyze importance of each block in each layer group.
        
        Args:
            dataloader: DataLoader with images
            num_samples: Number of samples to use
            
        Returns:
            Nested dict: {layer_group: {block_idx: BlockAnalysis}}
        """
        results = {group: {} for group in self.LAYER_GROUPS}
        
        if self.verbose:
            print("\nAnalyzing block importance...")
        
        # We need to capture intermediate activations between layer groups
        # For simplicity, we'll compute BI scores using a representative batch
        
        sample_batch = None
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                sample_batch = batch[0].to(self.device)
            else:
                sample_batch = batch.to(self.device)
            break
        
        if sample_batch is None:
            raise ValueError("Could not get sample batch from dataloader")
        
        self.model.eval()
        with torch.no_grad():
            # Forward through initial layers
            x = self.model.conv1(sample_batch)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            # Process each layer group
            for group in self.LAYER_GROUPS:
                layer = getattr(self.model, group)
                blocks = list(layer.children())
                
                for block_idx, block in enumerate(blocks):
                    # Compute BI for this block
                    bi_score = self._compute_block_bi(block, x)
                    
                    # Determine if redundant (low BI, not first/last block)
                    is_redundant = (
                        bi_score < 0.15 and
                        block_idx > 0 and
                        block_idx < len(blocks) - 1
                    )
                    
                    results[group][block_idx] = BlockAnalysis(
                        layer_group=group,
                        block_idx=block_idx,
                        bi_score=bi_score,
                        is_redundant=is_redundant
                    )
                    
                    # Forward through block for next iteration
                    x = block(x)
        
        if self.verbose:
            print("\n📊 Block Importance Summary:")
            for group in self.LAYER_GROUPS:
                scores = [r.bi_score for r in results[group].values()]
                redundant = sum(1 for r in results[group].values() if r.is_redundant)
                print(f"  {group}: BI range [{min(scores):.3f}, {max(scores):.3f}], "
                      f"{redundant} redundant blocks")
        
        return results
    
    def remove_block(
        self,
        layer_group: str,
        block_idx: int,
        inplace: bool = False
    ) -> nn.Module:
        """
        Remove a specific block from a layer group.
        
        Args:
            layer_group: 'layer1', 'layer2', 'layer3', or 'layer4'
            block_idx: Index of block to remove
            inplace: Whether to modify model in-place
            
        Returns:
            Model with block removed
        """
        if layer_group not in self.LAYER_GROUPS:
            raise ValueError(f"Invalid layer_group: {layer_group}")
        
        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)
        
        layer = getattr(model, layer_group)
        blocks = list(layer.children())
        
        if block_idx >= len(blocks):
            raise ValueError(
                f"block_idx {block_idx} out of range for {layer_group} "
                f"(has {len(blocks)} blocks)"
            )
        
        # Create new Sequential without the removed block
        new_blocks = nn.Sequential(*[
            b for i, b in enumerate(blocks) if i != block_idx
        ])
        
        setattr(model, layer_group, new_blocks)
        
        if self.verbose:
            print(f"✓ Removed block {block_idx} from {layer_group}")
            print(f"  {layer_group} now has {len(new_blocks)} blocks")
        
        if inplace:
            self.block_counts[layer_group] -= 1
        
        return model
    
    def remove_blocks(
        self,
        blocks_to_remove: List[Tuple[str, int]],
        inplace: bool = False
    ) -> nn.Module:
        """
        Remove multiple blocks.
        
        Args:
            blocks_to_remove: List of (layer_group, block_idx) tuples
            inplace: Whether to modify model in-place
            
        Returns:
            Model with blocks removed
        """
        if inplace:
            model = self.model
        else:
            model = copy.deepcopy(self.model)
        
        # Group by layer and sort blocks in reverse order (to maintain indices)
        by_layer = {}
        for layer_group, block_idx in blocks_to_remove:
            if layer_group not in by_layer:
                by_layer[layer_group] = []
            by_layer[layer_group].append(block_idx)
        
        for layer_group, block_indices in by_layer.items():
            layer = getattr(model, layer_group)
            blocks = list(layer.children())
            
            # Remove blocks (sorted in reverse to preserve indices)
            new_blocks = [
                b for i, b in enumerate(blocks)
                if i not in block_indices
            ]
            
            setattr(model, layer_group, nn.Sequential(*new_blocks))
            
            if self.verbose:
                print(f"✓ Removed {len(block_indices)} blocks from {layer_group}")
        
        return model
    
    def get_redundant_blocks(
        self,
        analysis: Dict[str, Dict[int, BlockAnalysis]]
    ) -> List[Tuple[str, int]]:
        """Get list of redundant blocks as (layer_group, block_idx) tuples."""
        redundant = []
        for group, blocks in analysis.items():
            for block_idx, result in blocks.items():
                if result.is_redundant:
                    redundant.append((group, block_idx))
        return redundant
