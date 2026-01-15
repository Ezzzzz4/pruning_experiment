# Understanding Layer Redundancy in Neural Networks: A Cross-Architecture Study


## Abstract

I conducted a systematic study of layer redundancy across language models and chain-of-thought reasoning models. Through 220+ benchmark evaluations, I found that many models contain significant redundancy in their middle layers—but not in the way previous Block Influence (BI) metrics suggested. Most surprisingly, I observed that removing specific layers can actually *improve* performance in certain tasks, challenging common assumptions about network depth.

**Key Results:**
- DeepSeek-R1-Distill improved by 10% on GSM8K after removing one layer
- Language models showed 67-86% of layers as "redundant" by BI scores, yet performance degraded rapidly
- Different capabilities (reasoning, factual recall, arithmetic) showed markedly different sensitivity to pruning

---

## 1. Introduction and Motivation

### The Problem

Modern neural networks are growing exponentially. GPT-4 reportedly uses 1.76 trillion parameters across many layers. This growth raises fundamental questions: Are all these layers necessary? Can we safely remove some to reduce inference costs?

Previous work (Men et al., 2024) suggested that middle layers in LLMs show high redundancy based on Block Influence scores. However, I noticed a gap: most studies measured statistical importance but didn't thoroughly validate whether models maintain *task performance* after layer removal.

### Research Questions

I set out to answer three specific questions:

1. **How does task performance degrade as we remove layers?** (Not just importance scores—actual accuracy on benchmarks)
2. **Do different model architectures show similar redundancy patterns?** (Language vs. reasoning models)
3. **Can we identify safe pruning strategies?** (Practical guidelines for model deployment)

### Scope

I focused on smaller models (< 2B parameters) where I could afford to run comprehensive benchmarks. I tested:
- **Language Models:** GPT-2 (124M), Qwen2.5-0.5B (500M), Qwen2.5-1.5B-Instruct (1.5B), TinyLlama-1.1B
- **Reasoning Models:** DeepSeek-R1-Distill-Qwen-1.5B (1.5B), Qwen2.5-Math-1.5B-Instruct (1.5B)

All experiments used greedy decoding for reproducibility.

---

## 2. Methodology

### 2.1 The Universal Handler: Auto-Discovery Design

Rather than hardcode patterns for each model architecture, I designed a **universal layer detection system** that automatically discovers prunable components:

```
┌─────────────────────────────────────────┐
│           UniversalHandler              │
│                                         │
│  1. Traverse model graph recursively    │
│  2. Identify nn.ModuleList instances    │
│  3. Classify by naming patterns:        │
│     - 'encoder', 'decoder'              │
│     - 'vision', 'text', 'audio'         │
│     - 'layers', 'blocks', 'h'           │
│  4. Store ComponentInfo for each        │
└─────────────────────────────────────────┘
```

**Pattern Matching:** The handler uses keyword matching to classify components:

```python
COMPONENT_PATTERNS = {
    'encoder': ['encoder', 'encode', 'enc_'],
    'decoder': ['decoder', 'decode', 'dec_'],
    'vision':  ['visual', 'vision', 'vit'],
    'text':    ['text', 'language', 'lm_'],
    'audio':   ['audio', 'speech', 'wav'],
}
```

**Usage Example:**
```python
from src.handlers import UniversalHandler

handler = UniversalHandler(model, verbose=True)
# Output: Discovered 1 component(s): main (28 layers at 'model.layers')

handler.remove_layers('main', [3, 4, 5], inplace=True)
# Directly removes layers 3-5 from the model
```

This approach works with GPT-2, LLaMA, Qwen, DeepSeek, TinyLlama, BERT, T5, Whisper, and many others—without any architecture-specific code.

### 2.2 Compound Benchmark Architecture

I built a **plugin-based benchmark system** that makes it easy to add new evaluation tasks:

```
experiments/benchmarks/
├── __init__.py      # Auto-discovery registry
├── base.py          # Abstract BaseBenchmark class
├── gsm8k.py         # GSM8K reasoning benchmark
└── language.py      # Language model perplexity benchmark
```

**Plugin Interface:**
```python
class BaseBenchmark(ABC):
    name: str           # Identifier (e.g., 'gsm8k')
    model_type: str     # 'reasoning', 'language', 'vision', 'audio'
    
    @abstractmethod
    def load_dataset(self, num_samples: Optional[int]) -> Dataset
    
    @abstractmethod
    def evaluate(self, model, tokenizer, dataset) -> Tuple[float, Dict]
    
    @abstractmethod
    def extract_answer(self, response: str) -> Any
```

**Auto-Discovery:** New benchmarks are automatically registered when placed in `experiments/benchmarks/`:

```python
# In __init__.py
BENCHMARKS = {}
for file in Path(__file__).parent.glob("*.py"):
    if file.stem not in ['__init__', 'base']:
        module = importlib.import_module(f".{file.stem}", package=__name__)
        for cls in module.__dict__.values():
            if isinstance(cls, type) and issubclass(cls, BaseBenchmark):
                BENCHMARKS[cls.name] = cls
```

### 2.3 Smart Pruning Algorithm

Rather than test all possible layer combinations, I implemented an automated stopping criterion:

```python
def smart_prune(model, benchmark, threshold=4, tolerance=0.05):
    """
    Remove layers one-by-one until N consecutive degradations.
    
    - threshold: Stop after N drops exceeding tolerance
    - tolerance: 5% drop considered "degradation"
    """
    baseline = evaluate(model)
    layers_removed = 0
    consecutive_drops = 0
    
    while consecutive_drops < threshold:
        layers_removed += 1
        remove_layer(model, layer_idx=2+layers_removed)
        score = evaluate(model)
        
        if (baseline - score) / baseline > tolerance:
            consecutive_drops += 1
        else:
            consecutive_drops = 0  # Reset if model recovers
    
    return results
```

This efficiently identifies each model's "breaking point" without exhaustive testing.

---

## 3. Results & Visual Analysis

### 3.1 Reasoning Models: The "Less is More" Phenomenon

I tested two 1.5B reasoning-specialized models on 50 GSM8K math problems. The results were remarkably consistent and counterintuitive:

#### DeepSeek-R1-Distill-Qwen-1.5B
![DeepSeek Pruning](results/figures/reasoning/deepseek_r1_distill_qwen_1_5b_pruning.png)

#### Qwen2.5-Math-1.5B-Instruct
![Qwen Math Pruning](results/figures/reasoning/qwen2_5_math_1_5b_instruct_pruning.png)

**Key Insight:** Both reasoning models improved or maintained performance when specific early layers were removed.
- **DeepSeek-R1-Distill:** Performance peaked after removing 1 layer (+10% accuracy).
- **Qwen2.5-Math:** Performance jumped (+6%) after removing 1 layer.

This suggests that for specialized reasoning tasks, certain model layers may be efficiently "skippable" or even detrimental, possibly acting as "noise" in the rigid logical path required for math problems.

### 3.2 Language Models

In contrast to reasoning models, general language models showed a starker sensitivity to pruning.

![Language Pruning Sensitivity](results/figures/language/benchmark_comparison.png)

**Task-Specific Degradation:**
- **GPT-2 (Blue):** Degrades immediately and linearly. Being a smaller model (124M), every layer is critical.
- **Qwen2.5-0.5B (Orange):** Shows resilience for the first few layers, then drops.
- **TinyLlama-1.1B (Green):** Surprisingly fragile despite its larger size, crashing after just 1-2 layers removed.

**The Control Group Conclusion (Qwen 1.5B):**
Crucially, I tested **Qwen2.5-1.5B-Instruct** (Red line)—the exact same base architecture and size as the Math model from Section 3.1.
- **Qwen Math 1.5B:** Gained +6% accuracy.
- **Qwen Language 1.5B:** **Lost -10% accuracy** after removing just one layer.

This confirms that "prunability" is not a function of model size, but of interaction between architecture and **training objective**.

### 3.3 Block Influence (BI) vs. Reality

I compared the Block Influence profiles of the Generalist (Language) and Specialist (Reasoning) models.

![Language vs Reasoning BI](results/figures/comparison/language_vs_reasoning_bi_comparison.png)

**The Visual Deception:**
As shown above, the BI profiles (the "bathtub" shapes) look remarkably similar. Both the Generalist (Blue) and Specialist (Orange/Green) models show low BI scores in their middle layers, suggesting high potential for pruning.

**The Reality Check:**
- **Reasoning Model:** The low BI scores were *accurate predictive signals*—the layers were indeed unnecessary.
- **Language Model:** The low BI scores were *false positives*—removing these "redundant" layers caused immediate performance collapse.

**Conclusion:** Block Influence measures **representation drift**, not **functional necessity**. A layer can change the representation very little (Low BI) but still be vital for maintaining the coherence of a general knowledge base.

---

## 4. Discussion

### 4.1 The Layer 2 Phenomenon

Across all three language models, **Layer 2** consistently showed disproportionately high Block Influence (BI) scores.

![Qwen Heatmap](results/figures/language/qwen2_5_0_5b_heatmap.png)

I hypothesize that Layer 2 represents a critical transition point where the model moves from simple token embeddings to initial semantic construction. Pruning this layer is almost always catastrophic.

### 4.2 Generalist vs. Specialist

The defining insight of this study comes from the direct comparison of the two Qwen 1.5B models.

- **Specialist Models (Reasoning/Math):** Behave like **Module Chains**.
    - They learn specific, rigid logic paths.
    - Redundant layers act as "noise" or "hesitation."
    - Pruning tightens the logic circuit, improving performance.

- **Generalist Models (Instruct/Chat):** Behave like **Holograms**.
    - Knowledge and coherence are distributed diffusely across the network.
    - Removing any "slice" (layer) degrades the resolution of the entire image.
    - There are no true "skip" connections for general knowledge.

**Implication for practitioners:** If you are deploying a specialized reasoning model, **you should give it a try to test layer pruning.** There is a non-trivial chance you MAY get a faster, smaller, and more accurate model. Nonetheless a further testing will be required.

### 4.4 Limitations and Caveats

I want to be clear about what this study *doesn't* cover:

**Model Size:** All tested models < 2B parameters. Results may not generalize to 7B+ models where emergent capabilities appear.

**Evaluation Method:** I used keyword matching and simple accuracy metrics. Human evaluation might reveal more nuanced degradation.

**Single Run:** No statistical significance testing across multiple random seeds.

**No Recovery Training:** I didn't attempt to fine-tune models after pruning. Performance might recover with continued training.

**English Only:** No multilingual or cross-lingual testing.

---

## 5. Practical Recommendations

Based on my findings, here's what I recommend for practitioners:

### For Model Deployment

| Scenario | Recommendation | Expected Impact |
|----------|----------------|-----------------|
| Edge deployment | Remove 1 layer from models with 20+ layers | 2-5% accuracy loss |
| Cost optimization | Test layer removal on YOUR specific task first | Highly task-dependent |
| Production systems | Don't remove layers without extensive validation | Risk of silent failures |

### For Researchers

1. **Don't trust BI scores alone** - Always validate with task benchmarks
2. **Test multiple tasks** - Pruning impact is highly task-dependent  
3. **Consider layer position** - Early and late layers are more critical
4. **Try "beneficial pruning"** - Some layers may hurt performance

---

## 6. Reproducing This Work

I've made all code and data publicly available. Here's how to run your own experiments:

### Installation

```bash
git clone https://github.com/yourusername/universal_model_pruning.git
cd universal_model_pruning
pip install -r requirements.txt
```

### Running Benchmarks

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

# Compute Block Influence scores
python experiments/analyze.py --model gpt2

# Generate figures
python experiments/visualize.py
```

### Data Organization

All experimental data is available in `results/`:

```
results/
├── data/
│   ├── reasoning/
│   │   └── deepseek_r1_distill_qwen_1_5b_gsm8k_benchmark.json
│   └── language/
│       ├── gpt2_comprehensive.json
│       ├── qwen2_5_0_5b_comprehensive.json
│       └── tinyllama_comprehensive.json
└── figures/
    └── [all visualizations]
```

---

## 7. Future Directions

This study opens several avenues for future research:

**Immediate Next Steps:**
- [ ] Test vision and audio models for cross-modality patterns

**Longer-Term Questions:**
- Can we train models to be inherently more prunable?
- Do different training objectives produce different redundancy patterns?
- Can we identify and remove redundant layers *during* training?
- Does the pattern generalize to larger models (7B+)?
- Can we restore perfomance by implementing recovery fine-tuning?
- Will the human evaluation provide more nuanced results?

---

## 8. Related Work

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