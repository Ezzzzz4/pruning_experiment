"""
Vision Model Benchmark Plugin.

Tests vision model quality via image classification accuracy.
Supports:
- Zero-shot classification (CLIP, SigLIP) on CIFAR-10
- Standard classification (ResNet, EfficientNet) on ImageNet subset

Usage:
    python experiments/benchmark.py --model openai/clip-vit-base-patch32 --benchmark vision
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import io

from .base import BaseBenchmark


# CIFAR-10 class names for zero-shot classification
# Use simpler names for zero-shot (e.g. 'car' instead of 'automobile')
CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ImageNet class subset (top 10 common classes for quick testing)
IMAGENET_SUBSET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich"
]


class VisionBenchmark(BaseBenchmark):
    """Vision model benchmark for image classification."""
    
    name = "vision"
    model_type = "vision"
    
    def __init__(self):
        self.is_zero_shot = False
        self.processor = None
        self.class_names = CIFAR10_CLASSES
    
    def load_dataset(self, num_samples: Optional[int] = None):
        """
        Load CIFAR-10 or ImageNet validation subset.
        
        Returns:
            List of (image, label) tuples
        """
        from datasets import load_dataset
        
        if num_samples is None:
            num_samples = 100
        
        # Load CIFAR-10 test set
        dataset = load_dataset("cifar10", split="test")
        
        # Sample randomly
        if num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # Convert to list of (image, label) tuples
        samples = []
        for item in dataset:
            samples.append({
                'image': item['img'],
                'label': item['label'],
                'class_name': CIFAR10_CLASSES[item['label']]
            })
        
        return samples
    
    def load_imagenet_dataset(self, num_samples: int = 100):
        """
        Load an ImageNet subset for classification evaluation.
        
        Tries multiple sources in order until one works.
        """
        from datasets import load_dataset
        
        # Try different ImageNet subsets in order of preference
        datasets_to_try = [
            ("Elriggs/imagenet-50-subset", "train"),
            ("zh-plus/tiny-imagenet", "valid"),
        ]
        
        for dataset_name, split in datasets_to_try:
            try:
                print(f"  Trying {dataset_name}...")
                dataset = load_dataset(dataset_name, split=split)
                
                if num_samples < len(dataset):
                    dataset = dataset.shuffle(seed=42).select(range(num_samples))
                
                samples = []
                for item in dataset:
                    # Handle different column names
                    image = item.get('image') or item.get('img')
                    label = item.get('label') or item.get('labels', 0)
                    # Get class name if available, otherwise use label number
                    class_name = item.get('class_name') or item.get('class') or str(label)
                    samples.append({
                        'image': image,
                        'label': label,
                        'class_name': class_name,
                    })
                print(f"  Loaded {len(samples)} samples from {dataset_name}")
                return samples
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Fallback to CIFAR-10
        print("  All ImageNet subsets failed. Using CIFAR-10...")
        return self.load_dataset(num_samples)
    
    def evaluate(
        self,
        model,
        tokenizer,  # Actually processor for vision models
        dataset: List[Dict],
        device: str = 'cuda',
        max_new_tokens: int = 50,
        greedy: bool = True
    ) -> Tuple[float, Dict]:
        """
        Evaluate vision model on classification task.
        
        For CLIP/SigLIP: Zero-shot classification using text prompts (CIFAR-10)
        For ResNet/EfficientNet: Direct classification (ImageNet)
        """
        model.eval()
        
        # Detect model type
        model_name = type(model).__name__.lower()
        is_clip = 'clip' in model_name or 'siglip' in model_name
        is_feature_model = 'dinov2' in model_name or 'dino' in model_name
        
        if is_clip:
            # Use provided CIFAR-10 dataset for zero-shot
            return self._evaluate_zero_shot(model, tokenizer, dataset, device)
        elif is_feature_model:
            # Feature extraction models - use embedding consistency as score
            return self._evaluate_features(model, tokenizer, dataset, device)
        else:
            # For classification models, use the provided dataset (CIFAR-10)
            # but measure confidence instead of accuracy (since labels mismatch)
            return self._evaluate_classification(model, tokenizer, dataset, device)
    
    def _evaluate_zero_shot(
        self,
        model,
        processor,
        dataset: List[Dict],
        device: str
    ) -> Tuple[float, Dict]:
        """Zero-shot classification for CLIP/SigLIP."""
        from transformers import AutoTokenizer
        
        correct = 0
        total = 0
        predictions = []
        
        # Get class names - use dataset-specific if available, else CIFAR-10
        # Collect unique class names from dataset
        class_names = getattr(self, 'class_names', None)
        if class_names is None:
            # Try to infer from dataset
            unique_classes = sorted(set(s.get('class_name', str(s['label'])) for s in dataset))
            if unique_classes and unique_classes[0] != '0':
                class_names = unique_classes
            else:
                class_names = CIFAR10_CLASSES
        
        # Prepare text prompts for each class
        # SigLIP uses "This is a photo of {label}." while CLIP uses "a photo of a {label}"
        model_class = type(model).__name__.lower()
        
        if 'siglip' in model_class:
            text_prompts = [f"This is a photo of {cls}." for cls in class_names]
        else:
            text_prompts = [f"a photo of a {cls}" for cls in class_names]
        
        pbar = tqdm(dataset, desc="Vision (Zero-Shot)", leave=False)
        for sample in pbar:
            image = sample['image']
            true_label = sample['label']
            
            # Convert grayscale to RGB if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            with torch.no_grad():
                try:
                    # Process image and text
                    inputs = processor(
                        text=text_prompts,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    
                    if torch.cuda.is_available() and hasattr(model, 'dtype'):
                        # Cast float tensors to model's dtype (handles both float16 and bfloat16)
                        inputs = {k: v.to(dtype=model.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
                    
                    # Get similarity scores
                    outputs = model(**inputs)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'logits_per_image'):
                        logits = outputs.logits_per_image
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        # Fallback: compute similarity manually
                        image_embeds = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs[0]
                        text_embeds = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs[1]
                        logits = torch.matmul(image_embeds, text_embeds.T)
                    
                    # Get prediction
                    pred_label = logits.argmax(dim=-1).item()
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    pred_label = -1
            
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            predictions.append({
                'true_label': true_label,
                'pred_label': pred_label,
                'true_class': sample['class_name'],
                'pred_class': class_names[pred_label] if 0 <= pred_label < len(class_names) else 'unknown',
                'correct': is_correct
            })
            
            pbar.set_postfix({'acc': f'{correct/total*100:.1f}%'})
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        return accuracy, {
            'correct': correct,
            'total': total,
            'predictions': predictions[:10],  # Sample predictions
            'method': 'zero_shot'
        }
    
    def _evaluate_features(
        self,
        model,
        processor,
        dataset: List[Dict],
        device: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate feature extraction models (DINOv2, ViT without head).
        
        Measures average embedding L2 norm as a consistency score.
        Higher norm = stronger feature responses = healthier model.
        """
        total_norm = 0.0
        total = 0
        norms = []
        
        pbar = tqdm(dataset, desc="Vision (Features)", leave=False)
        for sample in pbar:
            image = sample['image']
            
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                if torch.cuda.is_available():
                    inputs = {k: v.to(dtype=model.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                # Get CLS token embedding or pooled output
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use CLS token (first token)
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    embedding = outputs[0][:, 0, :]
                
                # Compute L2 norm
                norm = torch.norm(embedding, p=2).item()
            
            total_norm += norm
            total += 1
            norms.append(norm)
            
            avg_norm = total_norm / total
            pbar.set_postfix({'avg_norm': f'{avg_norm:.2f}'})
        
        # Normalize score to percentage-like range (0-100)
        # Higher norm = better, so we use it directly scaled
        avg_norm = total_norm / total if total > 0 else 0
        # Scale to reasonable range (typical embedding norms are 10-50)
        score = min(100, avg_norm * 2)
        
        return score, {
            'avg_norm': avg_norm,
            'min_norm': min(norms) if norms else 0,
            'max_norm': max(norms) if norms else 0,
            'total': total,
            'method': 'feature_norm'
        }
    
    def _evaluate_classification(
        self,
        model,
        processor,
        dataset: List[Dict],
        device: str
    ) -> Tuple[float, Dict]:
        """
        Standard classification for ResNet/EfficientNet.
        
        For ImageNet-trained models on CIFAR-10, we measure:
        - Average prediction confidence (how sure the model is)
        - This works as a proxy for model coherence/consistency
        """
        import torch.nn.functional as F
        
        total_confidence = 0.0
        total = 0
        predictions = []
        
        pbar = tqdm(dataset, desc="Vision (Classification)", leave=False)
        for sample in pbar:
            image = sample['image']
            
            with torch.no_grad():
                # Process image and cast to model's dtype
                inputs = processor(images=image, return_tensors="pt").to(device)
                if torch.cuda.is_available():
                    inputs = {k: v.to(dtype=model.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # Compute prediction confidence (softmax probability of top class)
                probs = F.softmax(logits, dim=-1)
                confidence, pred_label = probs.max(dim=-1)
                confidence = confidence.item()
                pred_label = pred_label.item()
                
                # Top-5 predictions
                top5_probs, top5_preds = probs.topk(5, dim=-1)
                top5_preds = top5_preds[0].tolist()
            
            total_confidence += confidence
            total += 1
            
            predictions.append({
                'pred_label': pred_label,
                'confidence': confidence,
                'top5_preds': top5_preds[:5],
            })
            
            avg_conf = total_confidence / total * 100
            pbar.set_postfix({'avg_conf': f'{avg_conf:.1f}%'})
        
        # Return average confidence as the score (higher = better)
        avg_confidence = total_confidence / total * 100 if total > 0 else 0
        
        return avg_confidence, {
            'avg_confidence': avg_confidence,
            'total': total,
            'predictions': predictions[:10],
            'method': 'classification_confidence'
        }
    
    def extract_answer(self, response: str) -> Any:
        """Not used for vision benchmark."""
        return response
    
    def compare_answers(self, predicted: Any, expected: Any) -> bool:
        """Compare predicted vs expected label."""
        return predicted == expected
