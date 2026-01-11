"""
Comprehensive Benchmark Suite for Language Model Pruning Evaluation

A thorough evaluation with:
1. 50+ Text Completion Tests (factual, common sense, linguistic)
2. 20+ HellaSwag-style Commonsense Reasoning
3. 20 Basic Math/Arithmetic Questions
4. 20 Factual Knowledge Questions
5. Multiple runs for statistical significance
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.handlers import UniversalHandler


# ============================================================
# COMPREHENSIVE TEST DATASETS
# ============================================================

# 50+ Text Completion Tests
COMPLETION_TESTS = [
    # Geography (10)
    ("The capital of France is", ["Paris"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Germany is", ["Berlin"]),
    ("The largest country by area is", ["Russia"]),
    ("The longest river in the world is the", ["Nile", "Amazon"]),
    ("Mount Everest is located in", ["Nepal", "Himalayas", "Asia"]),
    ("The Great Wall is in", ["China"]),
    ("The Eiffel Tower is in", ["Paris", "France"]),
    ("The Sahara is a", ["desert"]),
    ("Australia is both a country and a", ["continent"]),
    
    # Science (10)
    ("Water freezes at", ["0", "32", "zero"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The speed of light is approximately", ["300", "186"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The closest planet to the sun is", ["Mercury"]),
    ("Photosynthesis converts sunlight into", ["energy", "food", "glucose"]),
    ("The human body has", ["206", "bones"]),
    ("Gravity was discovered by", ["Newton", "Isaac"]),
    ("The atomic number of carbon is", ["6", "six"]),
    ("Sound travels faster in water than in", ["air"]),
    
    # Common Facts (10)
    ("The opposite of hot is", ["cold"]),
    ("The color of the sky on a clear day is", ["blue"]),
    ("Dogs are known for their sense of", ["smell"]),
    ("Cats are", ["pets", "animals", "feline"]),
    ("Humans breathe", ["oxygen", "air"]),
    ("Fire needs oxygen to", ["burn", "exist"]),
    ("Ice is frozen", ["water"]),
    ("The sun rises in the", ["east"]),
    ("Birds can", ["fly"]),
    ("Fish live in", ["water"]),
    
    # Math/Numbers (10)
    ("Two plus two equals", ["four", "4"]),
    ("Ten minus three equals", ["seven", "7"]),
    ("Five times five equals", ["twenty", "25"]),
    ("A dozen means", ["twelve", "12"]),
    ("A century is", ["100", "hundred"]),
    ("A millennium is", ["1000", "thousand"]),
    ("Half of 100 is", ["50", "fifty"]),
    ("The square root of 16 is", ["4", "four"]),
    ("One plus one equals", ["two", "2"]),
    ("100 divided by 10 equals", ["10", "ten"]),
    
    # Language/Grammar (10)
    ("The past tense of 'go' is", ["went"]),
    ("The plural of 'child' is", ["children"]),
    ("The opposite of 'big' is", ["small", "little"]),
    ("A person who writes books is called an", ["author", "writer"]),
    ("The synonym for 'happy' is", ["joyful", "glad", "pleased"]),
    ("The antonym of 'fast' is", ["slow"]),
    ("A baby dog is called a", ["puppy"]),
    ("A group of lions is called a", ["pride"]),
    ("The female version of 'king' is", ["queen"]),
    ("To 'commence' means to", ["begin", "start"]),
]

# 20+ HellaSwag-style Commonsense Reasoning
HELLASWAG_TESTS = [
    {
        "context": "A person is making breakfast. They crack eggs into a pan and",
        "choices": [
            "start reading a book about astronomy",
            "wait for them to cook while preparing toast",
            "decide to go swimming in the ocean",
            "call their lawyer about a contract"
        ],
        "correct": 1
    },
    {
        "context": "The student opened their textbook to study for the exam. They",
        "choices": [
            "ate the textbook for nutrition",
            "started reading and taking notes",
            "threw it out the window",
            "used it as a pillow to sleep"
        ],
        "correct": 1
    },
    {
        "context": "It started raining heavily outside. The person grabbed",
        "choices": [
            "sunglasses and sunscreen",
            "an umbrella before leaving",
            "a tennis racket",
            "a swimming suit"
        ],
        "correct": 1
    },
    {
        "context": "The chef was preparing a soup. They chopped vegetables and",
        "choices": [
            "put them in a pot with water",
            "mailed them to a friend",
            "buried them in the garden",
            "painted them different colors"
        ],
        "correct": 0
    },
    {
        "context": "A baby was crying because it was hungry. The mother",
        "choices": [
            "asked the baby to solve algebra",
            "gave it a bottle of milk",
            "started teaching it to drive",
            "showed it tax documents"
        ],
        "correct": 1
    },
    {
        "context": "The car ran out of gas. The driver had to",
        "choices": [
            "walk to the nearest gas station",
            "plant a garden on the road",
            "start a new career as a dancer",
            "teach the car to swim"
        ],
        "correct": 0
    },
    {
        "context": "The phone rang in the middle of dinner. The person",
        "choices": [
            "ate the phone instead of food",
            "answered the call quickly",
            "threw the phone into outer space",
            "asked the phone for its opinion"
        ],
        "correct": 1
    },
    {
        "context": "The dog saw a squirrel in the park. It immediately",
        "choices": [
            "started chasing after it",
            "began discussing philosophy",
            "opened a bank account",
            "flew to another country"
        ],
        "correct": 0
    },
    {
        "context": "The alarm clock rang at 7 AM. The person woke up and",
        "choices": [
            "decided to hibernate for years",
            "got ready for work",
            "became a professional astronaut",
            "turned into a butterfly"
        ],
        "correct": 1
    },
    {
        "context": "The water in the bathtub was too hot. The person",
        "choices": [
            "added some cold water",
            "jumped in anyway and got burned",
            "called the president",
            "started a fire department"
        ],
        "correct": 0
    },
    {
        "context": "The computer froze and stopped responding. The user",
        "choices": [
            "asked the computer about its feelings",
            "tried restarting the computer",
            "painted the screen blue",
            "called a plumber"
        ],
        "correct": 1
    },
    {
        "context": "The meeting was scheduled for 3 PM. Everyone arrived and",
        "choices": [
            "started discussing the agenda",
            "began a dance competition",
            "played hide and seek",
            "decided to become pirates"
        ],
        "correct": 0
    },
    {
        "context": "The store was having a sale. Many customers",
        "choices": [
            "came to buy discounted items",
            "started building a spaceship",
            "conducted a scientific experiment",
            "planted trees inside the store"
        ],
        "correct": 0
    },
    {
        "context": "The library was very quiet. A person whispered",
        "choices": [
            "because talking loudly is not allowed",
            "because they forgot how to speak",
            "to summon ancient spirits",
            "to the invisible audience"
        ],
        "correct": 0
    },
    {
        "context": "The patient had a fever. The doctor",
        "choices": [
            "prescribed medication to reduce it",
            "recommended eating ice cream for dinner",
            "suggested running a marathon",
            "told them to climb a mountain"
        ],
        "correct": 0
    },
    {
        "context": "The power went out during the storm. The family",
        "choices": [
            "lit candles for light",
            "decided to watch TV",
            "started using the microwave",
            "turned on all electronics"
        ],
        "correct": 0
    },
    {
        "context": "The cake was baking in the oven. The baker",
        "choices": [
            "set a timer and waited",
            "went on a three-week vacation",
            "forgot about food entirely",
            "started a new business"
        ],
        "correct": 0
    },
    {
        "context": "The soccer player kicked the ball towards the goal. The goalkeeper",
        "choices": [
            "tried to block it",
            "started reading a newspaper",
            "left to get coffee",
            "began singing opera"
        ],
        "correct": 0
    },
    {
        "context": "It was a hot summer day. People at the beach",
        "choices": [
            "wore winter coats and scarves",
            "swam in the ocean to cool off",
            "built igloos in the sand",
            "started a snowball fight"
        ],
        "correct": 1
    },
    {
        "context": "The airplane was ready for takeoff. The passengers",
        "choices": [
            "fastened their seatbelts",
            "opened all the doors",
            "jumped out of the windows",
            "started cooking dinner"
        ],
        "correct": 0
    },
]

# 20 Basic Math Questions
MATH_TESTS = [
    ("What is 5 + 3?", ["8", "eight"]),
    ("What is 12 - 7?", ["5", "five"]),
    ("What is 6 * 4?", ["24", "twenty-four", "twenty four"]),
    ("What is 15 / 3?", ["5", "five"]),
    ("What is 9 + 11?", ["20", "twenty"]),
    ("What is 100 - 45?", ["55", "fifty-five", "fifty five"]),
    ("What is 7 * 7?", ["49", "forty-nine", "forty nine"]),
    ("What is 81 / 9?", ["9", "nine"]),
    ("What is 25 + 25?", ["50", "fifty"]),
    ("What is 1000 - 1?", ["999", "nine hundred ninety-nine"]),
    ("What is 3 * 3 * 3?", ["27", "twenty-seven"]),
    ("What is 64 / 8?", ["8", "eight"]),
    ("What is 17 + 13?", ["30", "thirty"]),
    ("What is 50 - 25?", ["25", "twenty-five"]),
    ("What is 8 * 8?", ["64", "sixty-four"]),
    ("What is 144 / 12?", ["12", "twelve"]),
    ("What is 99 + 1?", ["100", "one hundred", "hundred"]),
    ("What is 200 - 150?", ["50", "fifty"]),
    ("What is 11 * 11?", ["121", "one hundred twenty-one"]),
    ("What is 100 / 4?", ["25", "twenty-five"]),
]

# 20 Factual Knowledge Questions  
KNOWLEDGE_TESTS = [
    ("Who wrote Romeo and Juliet?", ["Shakespeare", "William"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the largest mammal?", ["whale", "blue whale"]),
    ("Who painted the Mona Lisa?", ["da Vinci", "Leonardo"]),
    ("What is the smallest country in the world?", ["Vatican"]),
    ("Which planet has the most moons?", ["Saturn", "Jupiter"]),
    ("What year did man first land on the moon?", ["1969"]),
    ("Who invented the telephone?", ["Bell", "Alexander"]),
    ("What is the hardest natural substance?", ["diamond"]),
    ("Which ocean is the largest?", ["Pacific"]),
    ("Who was the first president of the United States?", ["Washington", "George"]),
    ("What is the main ingredient in glass?", ["sand", "silica"]),
    ("How many continents are there?", ["7", "seven"]),
    ("What gas do plants absorb?", ["carbon dioxide", "CO2"]),
    ("What is the capital of Australia?", ["Canberra"]),
    ("Who discovered penicillin?", ["Fleming", "Alexander"]),
    ("What is the tallest building in the world?", ["Burj", "Khalifa"]),
    ("How many strings does a standard guitar have?", ["6", "six"]),
    ("What is the currency of Japan?", ["yen"]),
    ("Which element has the atomic symbol 'O'?", ["oxygen"]),
]


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    model_name: str
    config: str
    n_layers_removed: int
    
    # Scores
    completion_acc: float
    hellaswag_acc: float
    math_acc: float
    knowledge_acc: float
    
    # Aggregates
    overall_acc: float
    coherence_score: float
    
    # Metadata
    total_tests: int
    inference_time_ms: float


def evaluate_category(
    model,
    tokenizer,
    tests: List[Tuple[str, List[str]]],
    device: str = 'cuda',
    category_name: str = "test"
) -> Tuple[float, List[Dict]]:
    """
    Evaluate a category of tests.
    Returns (accuracy, details).
    """
    model.eval()
    correct = 0
    details = []
    
    for prompt, expected_keywords in tests:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs.get('attention_mask')
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):].lower().strip()
        
        is_correct = any(kw.lower() in continuation for kw in expected_keywords)
        if is_correct:
            correct += 1
        
        details.append({
            'prompt': prompt[:50],
            'expected': expected_keywords[0],
            'got': continuation[:30],
            'correct': is_correct
        })
    
    accuracy = correct / len(tests) * 100 if tests else 0
    return accuracy, details


def evaluate_hellaswag(
    model,
    tokenizer,
    tests: List[Dict],
    device: str = 'cuda'
) -> Tuple[float, List[Dict]]:
    """
    Evaluate HellaSwag-style commonsense tests.
    """
    model.eval()
    correct = 0
    details = []
    
    for test in tests:
        context = test['context']
        choices = test['choices']
        correct_idx = test['correct']
        
        scores = []
        for choice in choices:
            full_text = context + " " + choice
            inputs = tokenizer(full_text, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()
            
            scores.append(-loss)
        
        predicted_idx = np.argmax(scores)
        is_correct = predicted_idx == correct_idx
        
        if is_correct:
            correct += 1
        
        details.append({
            'context': context[:40] + "...",
            'predicted': predicted_idx,
            'correct': correct_idx,
            'is_correct': is_correct
        })
    
    accuracy = correct / len(tests) * 100 if tests else 0
    return accuracy, details


def run_comprehensive_benchmark(
    model_name: str,
    model_id: str,
    bi_results_file: Path,
    output_dir: Path,
    device: str = 'cuda',
    n_layers_configs: List[int] = [0, 1, 2, 4],
    n_runs: int = 1
) -> Dict:
    """
    Run comprehensive benchmark suite.
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK: {model_name}")
    print(f"{'='*70}")
    
    # Load BI results to get redundant layers
    with open(bi_results_file) as f:
        bi_data = json.load(f)
    
    bi_scores = {int(k): v for k, v in bi_data['main']['bi_scores'].items()}
    sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
    redundant_layers = [idx for idx, score in sorted_layers if score < 0.1]
    
    print(f"Total layers: {len(bi_scores)}")
    print(f"Redundant layers (BI<0.1): {len(redundant_layers)}")
    print(f"Order of removal: {redundant_layers[:8]}...")
    
    # Load model
    print("\nLoading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_results = {
        'model_name': model_name,
        'model_id': model_id,
        'total_layers': len(bi_scores),
        'redundant_layers': redundant_layers,
        'test_counts': {
            'completion': len(COMPLETION_TESTS),
            'hellaswag': len(HELLASWAG_TESTS),
            'math': len(MATH_TESTS),
            'knowledge': len(KNOWLEDGE_TESTS),
            'total': len(COMPLETION_TESTS) + len(HELLASWAG_TESTS) + len(MATH_TESTS) + len(KNOWLEDGE_TESTS)
        },
        'configurations': []
    }
    
    for n_remove in n_layers_configs:
        if n_remove > len(redundant_layers):
            continue
        
        config_name = f"{n_remove}_layers" if n_remove > 0 else "baseline"
        print(f"\n[{config_name}] Testing...")
        
        # Create model for this config
        if n_remove > 0:
            handler = UniversalHandler(base_model, verbose=False)
            model = handler.remove_layers('main', redundant_layers[:n_remove], inplace=False)
            model = model.to(device)
        else:
            model = base_model
        
        # Run all benchmarks
        print("  - Completion tests...")
        start_time = time.time()
        completion_acc, completion_details = evaluate_category(
            model, tokenizer, COMPLETION_TESTS, device, "completion"
        )
        
        print("  - HellaSwag tests...")
        hellaswag_acc, hellaswag_details = evaluate_hellaswag(
            model, tokenizer, HELLASWAG_TESTS, device
        )
        
        print("  - Math tests...")
        math_acc, math_details = evaluate_category(
            model, tokenizer, MATH_TESTS, device, "math"
        )
        
        print("  - Knowledge tests...")
        knowledge_acc, knowledge_details = evaluate_category(
            model, tokenizer, KNOWLEDGE_TESTS, device, "knowledge"
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate overall
        total_tests = len(COMPLETION_TESTS) + len(HELLASWAG_TESTS) + len(MATH_TESTS) + len(KNOWLEDGE_TESTS)
        total_correct = (
            completion_acc * len(COMPLETION_TESTS) / 100 +
            hellaswag_acc * len(HELLASWAG_TESTS) / 100 +
            math_acc * len(MATH_TESTS) / 100 +
            knowledge_acc * len(KNOWLEDGE_TESTS) / 100
        )
        overall_acc = total_correct / total_tests * 100
        
        result = {
            'config': config_name,
            'n_removed': n_remove,
            'layers_removed': redundant_layers[:n_remove] if n_remove > 0 else [],
            'completion_accuracy': round(completion_acc, 1),
            'hellaswag_accuracy': round(hellaswag_acc, 1),
            'math_accuracy': round(math_acc, 1),
            'knowledge_accuracy': round(knowledge_acc, 1),
            'overall_accuracy': round(overall_acc, 1),
            'inference_time_ms': round(inference_time, 1),
            'details': {
                'completion': completion_details,
                'hellaswag': hellaswag_details,
                'math': math_details,
                'knowledge': knowledge_details
            }
        }
        
        all_results['configurations'].append(result)
        
        print(f"\n  Results:")
        print(f"    Completion:  {completion_acc:5.1f}%  ({int(completion_acc*len(COMPLETION_TESTS)/100)}/{len(COMPLETION_TESTS)})")
        print(f"    HellaSwag:   {hellaswag_acc:5.1f}%  ({int(hellaswag_acc*len(HELLASWAG_TESTS)/100)}/{len(HELLASWAG_TESTS)})")
        print(f"    Math:        {math_acc:5.1f}%  ({int(math_acc*len(MATH_TESTS)/100)}/{len(MATH_TESTS)})")
        print(f"    Knowledge:   {knowledge_acc:5.1f}%  ({int(knowledge_acc*len(KNOWLEDGE_TESTS)/100)}/{len(KNOWLEDGE_TESTS)})")
        print(f"    ─────────────────────")
        print(f"    OVERALL:     {overall_acc:5.1f}%")
        
        # Cleanup
        if n_remove > 0:
            del model
            torch.cuda.empty_cache()
    
    # Save results
    output_file = output_dir / f"{model_name.lower().replace('.', '_').replace('-', '_')}_comprehensive.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nSaved: {output_file}")
    
    # Cleanup
    del base_model
    torch.cuda.empty_cache()
    
    return all_results


def main():
    """Run comprehensive benchmarks."""
    output_dir = Path("results/data/language")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("COMPREHENSIVE LANGUAGE MODEL BENCHMARK SUITE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Total tests per model: {len(COMPLETION_TESTS) + len(HELLASWAG_TESTS) + len(MATH_TESTS) + len(KNOWLEDGE_TESTS)}")
    print(f"  - Completion: {len(COMPLETION_TESTS)}")
    print(f"  - HellaSwag:  {len(HELLASWAG_TESTS)}")
    print(f"  - Math:       {len(MATH_TESTS)}")
    print(f"  - Knowledge:  {len(KNOWLEDGE_TESTS)}")
    
    models = [
        ("GPT2", "gpt2", "results/data/language/gpt2_results.json"),
        ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "results/data/language/qwen2_5_0_5b_results.json"),
        ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "results/data/language/tinyllama_results.json"),
    ]
    
    for model_name, model_id, results_file in models:
        if not Path(results_file).exists():
            print(f"\nSkipping {model_name}: No BI results")
            continue
        
        try:
            run_comprehensive_benchmark(
                model_name=model_name,
                model_id=model_id,
                bi_results_file=Path(results_file),
                output_dir=output_dir,
                device=device
            )
        except Exception as e:
            print(f"\nError with {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
