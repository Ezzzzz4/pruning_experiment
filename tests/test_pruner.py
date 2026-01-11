"""
Test script for Universal Layer Pruning

Tests the core functionality:
1. Layer detection patterns
2. Block Influence computation
3. Layer removal
4. Benchmarking

Run with: python tests/test_pruner.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict


def test_layer_detection():
    """Test that layer detection works for common architectures."""
    print("\n" + "="*60)
    print("TEST 1: Layer Detection")
    print("="*60)
    
    try:
        from transformers import AutoModel
        from src.core.pruner import UniversalLayerPruner
        
        # Test GPT-2
        print("\n[1.1] Testing GPT-2 layer detection...")
        model = AutoModel.from_pretrained('gpt2')
        pruner = UniversalLayerPruner(model, task_type='language', verbose=True)
        
        assert pruner.layer_path == 'h', f"Expected 'h', got '{pruner.layer_path}'"
        assert pruner.n_layers == 12, f"Expected 12 layers, got {pruner.n_layers}"
        print("✓ GPT-2 detection passed")
        
        # Test BERT
        print("\n[1.2] Testing BERT layer detection...")
        model = AutoModel.from_pretrained('bert-base-uncased')
        pruner = UniversalLayerPruner(model, task_type='language', verbose=True)
        
        assert pruner.layer_path == 'encoder.layer', f"Expected 'encoder.layer', got '{pruner.layer_path}'"
        assert pruner.n_layers == 12, f"Expected 12 layers, got {pruner.n_layers}"
        print("✓ BERT detection passed")
        
        print("\n✅ Layer detection tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Layer detection test FAILED: {e}")
        return False


def test_block_influence():
    """Test Block Influence computation."""
    print("\n" + "="*60)
    print("TEST 2: Block Influence Computation")
    print("="*60)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        from src.core.block_influence import BlockInfluenceAnalyzer
        from torch.utils.data import DataLoader, Dataset
        
        # Load model
        print("\n[2.1] Loading GPT-2...")
        model = AutoModel.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create simple dataset
        class SimpleDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len=64):
                self.encodings = tokenizer(
                    texts, 
                    return_tensors='pt', 
                    padding='max_length',
                    truncation=True,
                    max_length=max_len
                )
            
            def __len__(self):
                return self.encodings['input_ids'].size(0)
            
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Neural networks can learn complex patterns.",
            "Deep learning models require lots of data.",
        ]
        
        dataset = SimpleDataset(texts, tokenizer)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Compute BI scores
        print("\n[2.2] Computing Block Influence scores...")
        analyzer = BlockInfluenceAnalyzer(model, device='cpu', verbose=True)
        
        layers = model.h
        bi_scores = analyzer.compute_bi_scores(dataloader, layers, 'h', num_samples=2)
        
        print(f"\nBI Scores:")
        for idx, score in sorted(bi_scores.items()):
            print(f"  Layer {idx}: {score:.4f}")
        
        # Verify scores are in valid range
        assert all(0 <= s <= 1 for s in bi_scores.values()), "BI scores should be in [0, 1]"
        assert len(bi_scores) == 12, f"Expected 12 layers, got {len(bi_scores)}"
        
        # Verify ranking works
        ranked = analyzer.rank_layers_by_importance(bi_scores)
        print(f"\nMost important layer: {ranked[0][0]} (BI={ranked[0][1]:.4f})")
        print(f"Least important layer: {ranked[-1][0]} (BI={ranked[-1][1]:.4f})")
        
        print("\n✅ Block Influence tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Block Influence test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_removal():
    """Test that layer removal produces valid model."""
    print("\n" + "="*60)
    print("TEST 3: Layer Removal")
    print("="*60)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        from src.core.pruner import UniversalLayerPruner
        
        # Load model
        print("\n[3.1] Loading GPT-2...")
        model = AutoModel.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        pruner = UniversalLayerPruner(model, task_type='language', verbose=False)
        
        print(f"Original layers: {pruner.n_layers}")
        
        # Remove layers 3, 4, 5
        print("\n[3.2] Removing layers 3, 4, 5...")
        pruned_model = pruner.remove_layers([3, 4, 5], inplace=False)
        
        # Get new layer count
        new_n_layers = len(pruned_model.h)
        print(f"Pruned layers: {new_n_layers}")
        
        assert new_n_layers == 9, f"Expected 9 layers after removal, got {new_n_layers}"
        
        # Verify model still works
        print("\n[3.3] Testing forward pass...")
        pruned_model = pruned_model.cpu()  # Ensure on CPU for testing
        inputs = tokenizer("Hello world", return_tensors='pt')
        
        with torch.no_grad():
            outputs = pruned_model(**inputs)
        
        assert outputs.last_hidden_state is not None
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        
        # Verify original model unchanged
        assert len(model.h) == 12, "Original model should be unchanged"
        
        print("\n✅ Layer removal tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Layer removal test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resnet_handler():
    """Test ResNet-specific handling."""
    print("\n" + "="*60)
    print("TEST 4: ResNet Handler")
    print("="*60)
    
    try:
        from torchvision.models import resnet18
        from src.handlers.resnet_handler import ResNetLayerPruner
        
        # Load model
        print("\n[4.1] Loading ResNet-18...")
        model = resnet18(pretrained=False)  # Don't download weights
        
        pruner = ResNetLayerPruner(model, verbose=True)
        
        print(f"\nBlock counts: {pruner.block_counts}")
        
        # Verify structure
        assert pruner.block_counts['layer1'] == 2
        assert pruner.block_counts['layer2'] == 2
        
        # Test block removal
        print("\n[4.2] Removing block 0 from layer3...")
        pruned = pruner.remove_block('layer3', 0, inplace=False)
        
        new_count = len(list(pruned.layer3.children()))
        print(f"layer3 now has {new_count} blocks")
        
        assert new_count == 1, f"Expected 1 block, got {new_count}"
        
        print("\n✅ ResNet handler tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ResNet handler test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistics():
    """Test statistical utilities."""
    print("\n" + "="*60)
    print("TEST 5: Statistical Utilities")
    print("="*60)
    
    try:
        from src.utils.statistics import (
            compute_confidence_interval,
            significance_test,
            cohens_d,
            compute_stats_summary
        )
        
        # Test data
        group_a = [1.2, 1.4, 1.1, 1.5, 1.3]
        group_b = [2.1, 2.3, 2.0, 2.4, 2.2]
        
        # Confidence interval
        print("\n[5.1] Testing confidence interval...")
        ci = compute_confidence_interval(group_a)
        print(f"95% CI for group_a: [{ci[0]:.3f}, {ci[1]:.3f}]")
        assert ci[0] < ci[1], "Lower bound should be less than upper bound"
        
        # Significance test
        print("\n[5.2] Testing significance test...")
        result = significance_test(group_a, group_b)
        print(f"t-test p-value: {result['p_value']:.4f}")
        print(f"Significant at 0.05: {result['significant_05']}")
        
        # Effect size
        print("\n[5.3] Testing Cohen's d...")
        d = cohens_d(group_a, group_b)
        print(f"Cohen's d: {d:.3f}")
        assert abs(d) > 0.8, "Effect should be large"
        
        # Summary stats
        print("\n[5.4] Testing stats summary...")
        summary = compute_stats_summary(group_a)
        print(f"Summary: mean={summary['mean']:.3f}, std={summary['std']:.3f}")
        
        print("\n✅ Statistics tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Statistics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("UNIVERSAL LAYER PRUNING - TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Run tests in order of dependency
    results['statistics'] = test_statistics()
    results['layer_detection'] = test_layer_detection()
    results['block_influence'] = test_block_influence()
    results['layer_removal'] = test_layer_removal()
    results['resnet_handler'] = test_resnet_handler()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
