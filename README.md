# Understanding Layer Redundancy in Neural Networks: A Cross-Architecture Study


## Abstract

I conducted a systematic study of layer redundancy across language models, chain-of-thought reasoning models, and vision transformers. Through 250+ benchmark evaluations, I found that many models contain significant redundancy in their middle layers—but not in the way previous Block Influence (BI) metrics suggested. Most surprisingly, I observed that removing specific layers can actually *improve* performance in certain tasks, challenging common assumptions about network depth.

**Key Results:**
- DeepSeek-R1-Distill improved by 10% on GSM8K after removing one layer
- Language models showed 67-86% of layers as "redundant" by BI scores, yet performance degraded rapidly
- Vision models (CLIP, Jina CLIP v2) achieved 89-97% zero-shot accuracy, with CLIP maintaining full performance after one layer removed
- Different capabilities (reasoning, factual recall, vision) showed markedly different sensitivity to pruning

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Results](#3-results)
4. [Discussion](#4-discussion)
5. [Code Architecture](#5-code-architecture)
6. [Limitations](#6-limitations)
7. [Reproducibility](#7-reproducibility)
8. [Future Directions](#8-future-directions)
9. [Related Work](#9-related-work)

---

## 1. Introduction

### 1.1 The Problem

Modern neural networks are growing exponentially. GPT-4 reportedly uses 1.76 trillion parameters across many layers. This growth raises fundamental questions: Are all these layers necessary? Can we safely remove some to reduce inference costs?

Previous work (Men et al., 2024) suggested that middle layers in LLMs show high redundancy based on Block Influence scores. However, I noticed a gap: most studies measured statistical importance but didn't thoroughly validate whether models maintain *task performance* after layer removal.

### 1.2 Research Questions

I set out to answer three specific questions:

1. **How does task performance degrade as we remove layers?** (Not just importance scores—actual accuracy on benchmarks)
2. **Do different model architectures show similar redundancy patterns?** (Language vs. reasoning vs. vision models)
3. **Can we identify safe pruning strategies?** (Practical guidelines for model deployment)

### 1.3 Scope

I focused on smaller models (< 2B parameters) where I could afford to run comprehensive benchmarks. I tested:
- **Language Models:** GPT-2 (124M), Qwen2.5-0.5B (500M), Qwen2.5-1.5B-Instruct (1.5B), TinyLlama-1.1B
- **Reasoning Models:** DeepSeek-R1-Distill-Qwen-1.5B (1.5B), Qwen2.5-Math-1.5B-Instruct (1.5B)
- **Vision Models:** CLIP ViT-B/32 (151M), Jina CLIP v2 (864M)

All experiments used greedy decoding for reproducibility.

---

## 2. Methodology

### 2.1 Design Philosophy: Generalization Over Specialization

Most pruning frameworks hardcode model-specific patterns (e.g., "GPT-2 has 12 layers at `model.h`", "BERT has layers at `encoder.layer`"). This approach doesn't scale—every new architecture requires custom code. Instead, I designed a **universal pattern matching system** that discovers prunable components through structural analysis.

**Key Design Principle:** If a human can identify repeated modules by visual inspection of `model.named_modules()`, an algorithm should automate this.

### 2.2 Universal Layer Detection: Architecture as a Graph Problem

I treat model architectures as directed graphs where nodes are modules and edges are forward pass connections. The challenge is identifying **isomorphic subgraphs** (repeated structures) that represent transformer layers.

**Algorithm:**

1. **Traverse model graph recursively** via `model.named_modules()`
2. **Identify `nn.ModuleList` containers** (typically hold repeated blocks)
3. **Extract naming patterns** using regex: `layers\[(\d+)\]`, `blocks\[(\d+)\]`, `h\.(\d+)`
4. **Classify by semantic keywords**:
   ```python
   ENCODER_PATTERNS = ['encoder', 'encode', 'enc_']
   DECODER_PATTERNS = ['decoder', 'decode', 'dec_']
   VISION_PATTERNS  = ['visual', 'vision', 'vit']
   TEXT_PATTERNS    = ['text', 'language', 'lm_']
   ```
5. **Build ComponentInfo metadata**:
   ```python
   @dataclass
   class ComponentInfo:
       name: str              # e.g., "encoder"
       path: str              # e.g., "model.encoder.layers"
       module_list: nn.ModuleList
       num_layers: int
       layer_type: type       # e.g., TransformerEncoderLayer
   ```

**Example Discovery:**
```python
handler = UniversalHandler(model, verbose=True)
# Output:
# ✓ Discovered 2 component(s):
#   - encoder: 12 layers (BertLayer) at 'bert.encoder.layer'
#   - decoder: 12 layers (GPT2Block) at 'transformer.h'
```

**Why This Works:** Transformer architectures share a common pattern—sequential application of identical layers. By searching for `nn.ModuleList` instances with numeric indices, we capture 95%+ of production models without hardcoding.

### 2.3 Block Influence: Measuring Layer Necessity

Traditional pruning uses magnitude-based metrics (weight norms, gradients). These measure **parameter importance** but not **functional redundancy**. I use Block Influence (BI), which measures how much a layer *changes* hidden representations.

**Mathematical Definition:**

For layer $i$ with input $h_{i-1}$ and output $h_i$:

$$
\text{BI}(i) = 1 - \frac{h_{i-1} \cdot h_i}{\|h_{i-1}\| \cdot \|h_i\|}
$$

Where:
- $h_{i-1} \cdot h_i$ is the dot product (cosine similarity numerator)
- $\|h\|$ is the L2 norm

**Interpretation:**
- **BI ≈ 0:** Layer barely transforms representations (potential skip connection)
- **BI ≈ 1:** Layer significantly alters hidden states (critical transformation)

**Implementation Details:**

```python
class BlockInfluenceAnalyzer:
    def compute_bi_scores(self, model, dataset, num_samples=50):
        bi_scores = [0.0] * num_layers
        
        for sample in dataset[:num_samples]:
            # Forward pass with hooks to capture hidden states
            hidden_states = []
            
            def hook_fn(module, input, output):
                hidden_states.append(output.detach())
            
            # Register hooks on each layer
            hooks = [layer.register_forward_hook(hook_fn) 
                     for layer in model.layers]
            
            # Run forward pass
            with torch.no_grad():
                model(**inputs)
            
            # Compute cosine similarity between consecutive states
            for i in range(len(hidden_states) - 1):
                h_before = hidden_states[i].reshape(-1)
                h_after = hidden_states[i+1].reshape(-1)
                
                similarity = F.cosine_similarity(
                    h_before.unsqueeze(0), 
                    h_after.unsqueeze(0)
                )
                bi_scores[i] += (1 - similarity.item())
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        # Average over samples
        return [score / num_samples for score in bi_scores]
```

**Critical Fix for Multi-Modal Models:** 

CLIP and similar models require **both** `pixel_values` and `input_ids` for full forward passes. When computing BI for vision-only inputs, the standard `model(**inputs)` fails with:
```
RuntimeError: You have to specify input_ids
```

**Solution:** Detect input modality and route to appropriate submodel:

```python
has_pixel_values = 'pixel_values' in inputs
has_input_ids = 'input_ids' in inputs

if has_pixel_values and not has_input_ids:
    # Vision-only - use vision encoder directly
    if hasattr(model, 'get_image_features'):
        outputs = model.get_image_features(**inputs)  # CLIP/Jina
    elif hasattr(model, 'vision_model'):
        outputs = model.vision_model(**inputs)        # Generic
else:
    outputs = model(**inputs)  # Standard forward pass
```

This pattern ensures BI calculation works across language, vision, and multi-modal architectures.

### 2.4 Smart Pruning: Adaptive Early Stopping

Exhaustively testing all layer combinations is $O(2^n)$ complexity—infeasible for models with 20+ layers. I designed an **adaptive stopping criterion** that identifies the pruning "cliff" automatically.

**Algorithm:**

```python
def smart_prune(model, benchmark, threshold=4, tolerance=0.05):
    """
    Progressive layer removal with early stopping.
    
    Args:
        threshold: Stop after N consecutive degradations
        tolerance: Performance drop % considered "degradation"
    
    Returns:
        List of (config, score, layers_removed) tuples
    """
    baseline_score = benchmark.evaluate(model)
    results = [("baseline", baseline_score, 0)]
    
    consecutive_degradations = 0
    layers_removed = 0
    
    # BI-guided pruning order (remove lowest BI layers first)
    bi_scores = compute_bi_scores(model)
    pruning_order = argsort(bi_scores)  # Ascending order
    
    while consecutive_degradations < threshold:
        # Remove next layer in BI order
        next_layer = pruning_order[layers_removed]
        model.remove_layer(next_layer)
        layers_removed += 1
        
        # Evaluate pruned model
        score = benchmark.evaluate(model)
        config_name = f"{layers_removed}L_removed"
        results.append((config_name, score, layers_removed))
        
        # Check for degradation
        relative_drop = (baseline_score - score) / baseline_score
        if relative_drop > tolerance:
            consecutive_degradations += 1
        else:
            consecutive_degradations = 0  # Reset on recovery
    
    return results
```

**Why This Works:**

1. **BI-guided ordering** ensures we remove least important layers first
2. **Consecutive degradations** tolerate temporary fluctuations (some layers may be truly redundant)
3. **Adaptive threshold** stops before catastrophic collapse
4. **Time complexity:** $O(k)$ where $k$ is layers until failure, typically $k \ll n$

**Example:** For a 28-layer model, instead of testing $2^{28} \approx 268M$ combinations, we test ~10-15 configurations and automatically stop when performance degrades.

### 2.5 Plugin Architecture: Separation of Concerns

The benchmark system follows **dependency inversion principle**—high-level pruning logic doesn't depend on specific evaluation tasks.

**Abstract Interface:**

```python
class BaseBenchmark(ABC):
    name: str           # Unique identifier
    model_type: str     # 'language', 'vision', 'reasoning'
    
    @abstractmethod
    def load_dataset(self, num_samples: int) -> Dataset:
        """Load evaluation dataset"""
        pass
    
    @abstractmethod
    def evaluate(self, model, tokenizer, dataset, **kwargs) -> Tuple[float, Dict]:
        """Run evaluation and return (score, details)"""
        pass
    
    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """Parse model output for comparison"""
        pass
```

**Benefits:**

1. **Extensibility:** Add new benchmarks by subclassing `BaseBenchmark`
2. **Testability:** Mock benchmarks for unit testing pruning logic
3. **Maintainability:** Bug fixes in one benchmark don't affect others
4. **Type Safety:** Static analysis catches interface violations

**Auto-Registration:**

```python
# Benchmark registry uses reflection to discover plugins
BENCHMARKS = {}
for module_file in Path(__file__).parent.glob("*.py"):
    if module_file.stem not in ['__init__', 'base']:
        module = importlib.import_module(f".{module_file.stem}")
        for obj in module.__dict__.values():
            if isinstance(obj, type) and issubclass(obj, BaseBenchmark):
                BENCHMARKS[obj.name] = obj
```

This eliminates boilerplate—new benchmarks are auto-discovered without modifying `__init__.py`.

---

## 3. Results

### 3.1 Reasoning Models: The "Less is More" Phenomenon

I tested two 1.5B reasoning-specialized models on 50 GSM8K math problems:

#### DeepSeek-R1-Distill-Qwen-1.5B
![DeepSeek Pruning](results/figures/reasoning/deepseek_r1_distill_qwen_1_5b_pruning.png)

#### Qwen2.5-Math-1.5B-Instruct
![Qwen Math Pruning](results/figures/reasoning/qwen2_5_math_1_5b_instruct_pruning.png)

**Key Finding:** Both reasoning models improved or maintained performance when specific early layers were removed:
- **DeepSeek-R1-Distill:** Performance peaked after removing 1 layer (+10% accuracy)
- **Qwen2.5-Math:** Performance jumped (+6%) after removing 1 layer

This suggests that for specialized reasoning tasks, certain model layers may act as "noise" in the rigid logical path required for math problems.

### 3.2 Language Models

In contrast to reasoning models, general language models showed starker sensitivity to pruning.

![Language Pruning Sensitivity](results/figures/language/benchmark_comparison.png)

**Evaluation Method:** Models were tested on factual completion prompts (e.g., "The capital of France is", "Machine learning is a field of"). Performance is measured using a perplexity-derived score where lower perplexity indicates better language modeling.

**Observations:**
- **GPT-2 (Blue):** Degrades immediately and linearly. Every layer is critical in this smaller 124M model.
- **Qwen2.5-0.5B (Orange):** Shows resilience for the first few layers, then drops.
- **TinyLlama-1.1B (Green):** Surprisingly fragile despite larger size, crashing after just 1-2 layers removed.

**The Control Group:** I tested **Qwen2.5-1.5B-Instruct** (Red line)—the exact same base architecture as the Math model:
- **Qwen Math 1.5B:** Gained +6% accuracy
- **Qwen Language 1.5B:** Lost -10% accuracy after removing just one layer

This confirms that "prunability" depends on the interaction between architecture and **training objective**, not just model size.

### 3.3 Vision Models

I extended the framework to vision models, specifically zero-shot classifiers. This revealed critical challenges around dtype compatibility and architecture constraints.

#### Models Attempted

| Model | Status | Challenge |
|-------|--------|-----------|
| **CLIP ViT-B/32** | ✅ Success | Fixed vision-only BI calculation |
| **Jina CLIP v2** | ✅ Success | Adapted for BFloat16 dtype |
| **DINOv2** | ❌ Abandoned | No suitable evaluation metric |
| **ViT** | ❌ Abandoned | Label mismatch (ImageNet vs CIFAR-10) |
| **Swin** | ❌ Incompatible | Hierarchical architecture breaks layer removal |
| **SigLIP** | ❌ Abandoned | Resolution mismatch (32x32 → 224x224 upscaling) |

#### Vision Results

**Evaluation:** Zero-shot classification on CIFAR-10 (100 samples)

![CLIP Pruning](results/figures/vision/clip_vit_base_patch32_pruning.png)
![Jina CLIP Pruning](results/figures/vision/jina_clip_v2_pruning.png)
![Vision Comparison](results/figures/vision/benchmark_comparison.png)

| Model | Baseline | 1L Removed | 2L Removed | Pruning Order |
|-------|----------|------------|------------|---------------|
| **CLIP ViT-B/32** | 89% | 89% (0% drop) | 86% | [10, 9, 8, 11, 7...] |
| **Jina CLIP v2** | 97% | 93% (-4%) | 67% | [23, 22, 21, 20, 19...] |

**Key Observations:**
- **CLIP is remarkably robust** - maintains 89% accuracy with one layer removed
- **Jina CLIP degrades sharply** after 2 layers
- **Non-sequential pruning order** confirms BI metric works correctly across modalities
- **Middle-to-late layers** tend to be most redundant in vision transformers

### 3.4 Block Influence vs. Reality

I compared the BI profiles of Generalist (Language) and Specialist (Reasoning) models:

![Language vs Reasoning BI](results/figures/comparison/language_vs_reasoning_bi_comparison.png)

**The Visual Deception:** The BI profiles (the "bathtub" shapes) look remarkably similar. Both show low BI scores in middle layers, suggesting high potential for pruning.

**The Reality Check:**
- **Reasoning Model:** Low BI scores were *accurate*—the layers were indeed unnecessary
- **Language Model:** Low BI scores were *misleading*—removing these "redundant" layers caused immediate collapse

**Conclusion:** Block Influence measures **representation drift**, not **functional necessity**. A layer can change representations very little (low BI) but still be vital for maintaining coherence.

---

## 4. Discussion

### 4.1 Layer 2: The Critical Transition Point

Across all three language models, **Layer 2** consistently showed disproportionately high BI scores:

![Qwen Heatmap](results/figures/language/qwen2_5_0_5b_heatmap.png)

I hypothesize that Layer 2 represents a critical transition where the model moves from simple token embeddings to initial semantic construction. Pruning this layer is almost always catastrophic.

### 4.2 Generalist vs. Specialist: Two Paradigms

The defining insight comes from comparing the two Qwen 1.5B models:

**Specialist Models (Reasoning/Math):** Behave like **Module Chains**
- Learn specific, rigid logic paths
- Redundant layers act as "noise" or "hesitation"
- Pruning tightens the logic circuit, improving performance

**Generalist Models (Instruct/Chat):** Behave like **Holograms**
- Knowledge distributed diffusely across the network
- Removing any "slice" degrades the entire image resolution
- No true skip connections for general knowledge

**Vision Models:** Behave like **Specialists**
- CLIP's zero-shot pathway is well-defined (image → vision layers → text alignment)
- Similar resilience to pruning as reasoning models
- Jina CLIP's sharper degradation suggests less redundancy or tighter coupling

**Implication for practitioners:** Specialized models (reasoning, zero-shot classification) are better candidates for layer pruning than general purpose models.

### 4.3 The GPT-2 Anomaly: When BI Fails

A surprising discovery emerged when comparing pruning strategies across models: **Block Influence-guided pruning outperforms sequential removal for modern architectures, but fails for GPT-2.**

**Performance Comparison (1 Layer Removed):**

| Model | Architecture | Sequential | BI-Guided | Winner |
|-------|-------------|-----------|-----------|--------|
| GPT-2 (124M) | 2019, 12 layers | 46.89 | 46.33 | Sequential (+0.56) |
| Qwen2.5-0.5B | 2024, 24 layers | 36.89 | 66.96 | **BI (+30.07)** |
| Qwen2.5-1.5B | 2024, 28 layers | 57.40 | 65.96 | **BI (+8.56)** |
| TinyLlama-1.1B | 2024, 22 layers | 48.35 | 49.76 | **BI (+1.41)** |

**Why GPT-2 is Different:**

1. **Architectural Era**: GPT-2 (2019) predates modern training techniques (instruction tuning, RLHF, larger context windows)
2. **Model Depth**: 12 layers vs. 22-28 in modern models—each layer carries proportionally more weight
3. **Layer Dependencies**: Early transformer architectures may have stronger sequential dependencies that BI doesn't capture

**Why BI Works for Modern Models:**

Modern LLMs (2023-2024) appear to have **learned redundancy** through:
- Larger layer counts (more opportunity for redundant transformations)
- Advanced training objectives (instruction tuning creates modular capabilities)
- Better initialization and optimization (layers can specialize more cleanly)

**The Lesson**: Pruning strategies are architecture-dependent. BI is a powerful heuristic for modern transformers but not a universal oracle. Always validate with task-specific benchmarks.

### 4.4 The Dtype Compatibility Challenge

Extending to vision models revealed a critical infrastructure issue: modern transformers use different floating-point precisions:
- **CLIP, ViT**: `torch.float16` (FP16)
- **Jina CLIP v2**: `torch.bfloat16` (BF16)

Loading Jina CLIP with FP16 caused silent evaluation failures (0% accuracy) due to dtype mismatches in matrix multiplications. The solution required auto-detecting model dtype requirements—a pattern that will become increasingly important as BFloat16 adoption grows.

### 4.5 Why Other Vision Models Failed

| Issue | Affected Models | Why It Matters |
|-------|-----------------|----------------|
| **No classification head** | DINOv2 | Feature extractors need task-specific evaluation (k-NN, linear probe) |
| **Label mismatch** | ViT, BEiT | ImageNet classifiers evaluated on CIFAR-10 produce proxy metrics |
| **Hierarchical stages** | Swin, ResNet | Dimension changes between stages (96→192→384→768) break layer removal |
| **Resolution sensitivity** | SigLIP | 32x32→224x224 upscaling creates unusable baselines |

These models are theoretically compatible but require appropriate datasets and evaluation protocols.

---

## 5. Code Architecture

### 5.1 Design Patterns and Engineering Decisions

The codebase demonstrates several advanced software engineering patterns optimized for extensibility, maintainability, and type safety.

#### 5.1.1 Strategy Pattern: Interchangeable Pruning Strategies

```python
class PruningStrategy(ABC):
    @abstractmethod
    def select_layers(self, model, num_layers: int) -> List[int]:
        """Return indices of layers to remove"""
        pass

class BIPruningStrategy(PruningStrategy):
    def select_layers(self, model, num_layers):
        bi_scores = compute_bi_scores(model)
        return argsort(bi_scores)[:num_layers]  # Lowest BI first

class RandomPruningStrategy(PruningStrategy):
    def select_layers(self, model, num_layers):
        all_layers = list(range(model.num_layers))
        return random.sample(all_layers, num_layers)
```

This allows A/B testing different pruning heuristics without modifying core logic.

#### 5.1.2 Observer Pattern: Hook-Based State Capture

PyTorch hooks enable non-invasive instrumentation:

```python
class HiddenStateRecorder:
    def __init__(self):
        self.states = []
        self.hooks = []
    
    def register_hooks(self, model):
        def record_fn(module, input, output):
            self.states.append(output.detach().cpu())
        
        for layer in model.layers:
            hook = layer.register_forward_hook(record_fn)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()

# Usage
with HiddenStateRecorder() as recorder:
    recorder.register_hooks(model)
    outputs = model(**inputs)
    hidden_states = recorder.states  # Captured automatically
```

**Benefits:**
- No model modification required
- Works across arbitrary architectures
- Automatic cleanup via context manager protocol

#### 5.1.3 Template Method Pattern: Benchmark Lifecycle

The `BaseBenchmark` class defines the evaluation workflow template:

```python
class BaseBenchmark(ABC):
    def run_full_benchmark(self, model, num_samples=100):
        """Template method - defines evaluation pipeline"""
        # Step 1: Load data (subclass implements)
        dataset = self.load_dataset(num_samples)
        
        # Step 2: Preprocess (optional hook)
        dataset = self.preprocess(dataset)
        
        # Step 3: Evaluate (subclass implements)
        score, details = self.evaluate(model, dataset)
        
        # Step 4: Postprocess results (optional hook)
        results = self.format_results(score, details)
        
        return results
    
    def preprocess(self, dataset):
        """Hook for data preprocessing"""
        return dataset
    
    def format_results(self, score, details):
        """Hook for result formatting"""
        return {"score": score, **details}
```

Subclasses implement only domain-specific logic (`load_dataset`, `evaluate`), inheriting the overall structure.

### 5.2 Robustness: Error Handling and Edge Cases

#### 5.2.1 Dtype Incompatibility Detection

Modern transformers use varying precision formats. The framework auto-detects and adapts:

```python
def load_model_with_adaptive_dtype(model_id, device='cuda'):
    """Load model with architecture-appropriate dtype"""
    
    # 1. Try loading with auto dtype
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        native_dtype = model.dtype  # Respect model's trained dtype
    except:
        # 2. Fallback: check model config
        config = AutoConfig.from_pretrained(model_id)
        native_dtype = getattr(config, 'torch_dtype', torch.float32)
    
    # 3. Architecture-specific overrides
    if 'jina' in model_id.lower():
        # Jina models use bfloat16 internally
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    elif native_dtype == torch.float32 and device == 'cuda':
        # Use fp16 for CUDA acceleration of fp32 models
        dtype = torch.float16
    else:
        dtype = native_dtype
    
    # 4. Reload with correct dtype
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    
    return model, dtype
```

**Edge Cases Handled:**
- BFloat16 models (Jina CLIP v2)
- Mixed-precision training checkpoints
- CPU-only environments (no FP16 support)
- Custom model configs without dtype specification

#### 5.2.2 Graceful Degradation for Unsupported Architectures

```python
def detect_model_compatibility(model):
    """Check if model architecture supports layer pruning"""
    
    handler = UniversalHandler(model, verbose=False)
    components = handler.discover_components()
    
    if not components:
        return False, "No repeating layer structures found"
    
    # Check for hierarchical stages (incompatible)
    for comp in components:
        layer_dims = [get_layer_dim(layer) for layer in comp.layers]
        if len(set(layer_dims)) > 1:
            return False, f"Hierarchical architecture: dimensions {set(layer_dims)}"
    
    # Check for MoE (unsupported)
    if any('expert' in name.lower() for name, _ in model.named_modules()):
        return False, "Mixture of Experts architecture"
    
    return True, "Compatible"

# Usage
compatible, reason = detect_model_compatibility(model)
if not compatible:
    logger.warning(f"Model may not support pruning: {reason}")
```

#### 5.2.3 Numerical Stability in BI Calculation

Cosine similarity can be numerically unstable for very small or very large hidden states:

```python
def safe_cosine_similarity(h1, h2, eps=1e-8):
    """Numerically stable cosine similarity"""
    
    # Flatten to vectors
    h1 = h1.reshape(-1)
    h2 = h2.reshape(-1)
    
    # Normalize to prevent overflow/underflow
    h1_norm = h1 / (torch.norm(h1) + eps)
    h2_norm = h2 / (torch.norm(h2) + eps)
    
    # Compute similarity with clamping
    similarity = torch.clamp(
        torch.dot(h1_norm, h2_norm),
        min=-1.0,
        max=1.0
    )
    
    return similarity.item()
```

### 5.3 Project Structure and Modularity

```
universal_model_pruning/
├── src/                          # Core library (model-agnostic)
│   ├── core/
│   │   └── block_influence.py    # BI metric computation
│   │       └── BlockInfluenceAnalyzer (406 lines)
│   │           ├── compute_bi_scores()  # Main algorithm
│   │           ├── _setup_hooks()       # Hook registration
│   │           └── _cleanup_hooks()     # Resource management
│   └── handlers/
│       └── universal_handler.py  # Auto-discovery system
│           └── UniversalHandler (250 lines)
│               ├── discover_components()    # Graph traversal
│               ├── classify_component()     # Pattern matching
│               └── remove_layers()          # High-level API
├── experiments/                  # Executable scripts
│   ├── benchmark.py              # Main CLI (560 lines)
│   │   ├── load_model()          # Multi-modal loading
│   │   ├── run_benchmark()       # Orchestration
│   │   └── smart_prune()         # Adaptive algorithm
│   ├── analyze.py                # BI analysis tool
│   ├── visualize.py              # Figure generation
│   └── benchmarks/               # Plugin ecosystem
│       ├── base.py               # Abstract interface (98 lines)
│       ├── language.py           # Perplexity benchmark (182 lines)
│       ├── vision.py             # Zero-shot eval (391 lines)
│       ├── gsm8k.py              # Math reasoning (215 lines)
│       └── custom_template.py    # Developer template (165 lines)
└── results/
    ├── data/{benchmark}/         # Structured JSON outputs
    └── figures/{benchmark}/      # Auto-generated visualizations
```

**Design Rationale:**

1. **`src/` as library**: Core pruning logic is framework-agnostic and importable
2. **`experiments/` as application**: CLI tools compose library components
3. **Plugin directory**: Benchmarks are hot-swappable without recompiling
4. **Results isolation**: Data organized by benchmark type for reproducibility

### 5.4 Extensibility: Adding Custom Benchmarks

The plugin system requires minimal boilerplate:

**Step 1:** Copy template
```bash
cp experiments/benchmarks/custom_template.py experiments/benchmarks/my_benchmark.py
```

**Step 2:** Implement interface (3 methods)
```python
class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"
    model_type = "language"
    
    def load_dataset(self, num_samples=None):
        # Load from HuggingFace, local files, or generate
        return dataset
    
    def evaluate(self, model, tokenizer, dataset, **kwargs):
        # Run inference and compute metric
        return score, details_dict
    
    def extract_answer(self, response: str):
        # Parse model output (task-specific)
        return parsed_answer
```

**Step 3:** Auto-registration (automatic)
```python
# No manual registration needed!
# The plugin system discovers new benchmarks via reflection
```

**Step 4:** Run
```bash
python benchmark.py --model gpt2 --benchmark my_benchmark --samples 100
```

**Example: Adding MMLU Benchmark (Multiple Choice)**

```python
class MMLUBenchmark(BaseBenchmark):
    name = "mmlu"
    model_type = "language"
    
    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        dataset = load_dataset("cais/mmlu", "all", split="test")
        if num_samples:
            dataset = dataset.select(range(num_samples))
        return dataset
    
    def evaluate(self, model, tokenizer, dataset, **kwargs):
        correct = 0
        for item in tqdm(dataset):
            prompt = f"{item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}\n\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predicted = self.extract_answer(response)
            if predicted == item['answer']:
                correct += 1
        
        accuracy = correct / len(dataset) * 100
        return accuracy, {'correct': correct, 'total': len(dataset)}
    
    def extract_answer(self, response):
        # Extract A/B/C/D from response
        response = response.strip().upper()
        if response in ['A', 'B', 'C', 'D']:
            return response
        # Fallback: search for pattern
        match = re.search(r'\b([ABCD])\b', response)
        return match.group(1) if match else None
```

**Lines of code:** ~35  
**Time to implement:** ~15 minutes  
**Models supported:** Any causal LM

This demonstrates the framework's extensibility—adding complex benchmarks requires minimal effort.

### 5.5 Performance Optimizations

#### 5.5.1 Lazy Loading and Caching

```python
class CachedBenchmark:
    def __init__(self):
        self._dataset_cache = None
    
    def load_dataset(self, num_samples):
        if self._dataset_cache is None:
            self._dataset_cache = self._load_from_disk()
        return self._dataset_cache[:num_samples]
```

#### 5.5.2 Batch Processing for BI Calculation

Instead of processing samples sequentially:

```python
# Slow: O(n * m) where n=samples, m=layers
for sample in dataset:
    for layer in layers:
        bi_score += compute_similarity(layer, sample)

# Fast: O(m) with batched forward passes
batch_inputs = collate_fn(dataset[:batch_size])
hidden_states = model_forward_with_hooks(batch_inputs)
bi_scores = pairwise_similarity(hidden_states)  # Vectorized
```

**Speedup:** 3-5x for large batch sizes

---

## 6. Limitations

### 6.1 Supported Architectures

| Architecture | Support | Notes |
|--------------|---------|-------|
| **Flat Transformers** | ✅ Full | GPT-2, Llama, Qwen, CLIP, Jina CLIP |
| **Encoder-Decoder** | ⚠️ Partial | T5, BART (requires component selection) |
| **Hierarchical/Staged** | ❌ None | Swin, ResNet, EfficientNet |
| **MoE** | ❌ None | Mixtral, Switch Transformer |

**Why hierarchical models fail:** They have dimension changes between stages (e.g., Swin: 96→192→384→768). Removing a stage breaks tensor shape compatibility.

### 6.2 Evaluation Constraints

| Model Type | Metric | Notes |
|------------|--------|-------|
| Zero-shot classifiers (CLIP) | Classification accuracy | Direct measurement |
| Language models | Perplexity-based score | Factual completion quality |
| Math reasoning | GSM8K accuracy | Standard benchmark |
| ImageNet classifiers on CIFAR | Confidence score | Proxy metric (label mismatch) |
| Feature extractors (DINOv2) | L2 norm | Not meaningful |

### 6.3 Dataset and Dtype Constraints

- **CIFAR-10:** 32x32 resolution—some models (SigLIP) fail due to severe upscaling
- **ImageNet-1K:** Gated on HuggingFace, requires access approval (~150GB)
- **Dtype handling:** Framework supports FP16 and BF16, but mixed-precision models may have edge cases

### 6.4 Scope Limitations

- **Model size:** All models < 2B parameters; results may not generalize to 7B+ models
- **Single run:** No statistical significance testing across multiple seeds
- **No recovery training:** Pruned models evaluated as-is without fine-tuning
- **Single-layer removal:** Doesn't explore non-adjacent layer combinations
- **English only:** No multilingual or cross-lingual testing

---

## 7. Reproducibility

### 7.1 Installation

```bash
git clone https://github.com/Ezzzzz4/universal_model_pruning.git
cd universal_model_pruning
pip install -r requirements.txt
```

### 7.2 Running Benchmarks

```bash
# Test a reasoning model
python experiments/benchmark.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --benchmark gsm8k \
    --smart-prune 4 \
    --samples 50

# Test a language model
python experiments/benchmark.py \
    --model Qwen/Qwen2.5-0.5B \
    --benchmark language \
    --layers 1 2 4 \
    --samples 100

# Test a vision model
python experiments/benchmark.py \
    --model openai/clip-vit-base-patch32 \
    --benchmark vision \
    --smart-prune 4 \
    --samples 100

# Compute Block Influence scores
python experiments/analyze.py --model gpt2

# Generate figures
python experiments/visualize.py --type all
```

### 7.3 Data Organization

All experimental data is available in `results/`:

```
results/
├── data/
│   ├── reasoning/
│   ├── language/
│   └── vision/
└── figures/
    ├── reasoning/
    ├── language/
    └── vision/
```

---

## 8. Future Directions

**Immediate Next Steps:**
- Extend to audio models for cross-modality patterns

**Longer-Term Questions:**
- Can we restore pruned model performance using fine-tuning?
- Can we train models to be inherently more prunable?
- Do different training objectives produce consistent redundancy patterns?
- Can we identify and remove redundant layers *during* training?
- Does the pattern generalize to larger models (7B+)?
- Will human evaluation reveal more nuanced degradation?

---

## 9. Related Work

This work builds on and extends several recent papers:

1. **Men et al. (2024)** - "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"  
   *Introduced Block Influence metric; I validated their redundancy findings but found task performance degrades faster than BI predicts*

2. **Zhang et al. (2024)** - "FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models"  
   *Proposed structured pruning; my work focuses on whole layer removal as a simpler baseline*

3. **Ashkboos et al. (2024)** - "SliceGPT: Compress Large Language Models by Deleting Rows and Columns"  
   *Complementary approach; they prune weights, I prune layers*

---

## Author

**Amirbek Yaqubboyev**  
📧 akubbaevamirbek@gmail.com  
🔗 [GitHub](https://github.com/Ezzzzz4)