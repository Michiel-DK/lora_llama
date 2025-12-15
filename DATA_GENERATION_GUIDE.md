# Data Generation Guide

## Three Prompt Types

### 1. Balanced (Default) - `--prompt-type balanced`
Generates 4 variations per FLORES example with scores: 3-5, 6-7, 8-9, 10
- Best for general training
- Covers full quality spectrum
- **Current training data used this**

### 2. Low Scores - `--prompt-type low_scores` 
Generates 4 variations per FLORES example with scores: 1-2, 2-3, 3-4, 4-5
- Specifically for training model to recognize BAD translations
- Focuses on major semantic errors, wrong meanings, comprehension issues
- Addresses over-optimistic scoring (model giving 9/10 to bad translations)

### 3. Middle Range - `--prompt-type middle_range`
Generates 4 variations per FLORES example with scores: 4-5, 5-6, 6-7, 7-8
- **NEW**: Focuses on ambiguous/borderline cases
- Teaches distinction between "unacceptable" (≤5) and "acceptable" (≥6)
- Addresses model weakness in middle quality range (scores 4-7)
- Mix of moderate and minor errors

## Usage

### Generate low-score training data (recommended next step):
```bash
python pt_app/eval_model/judge_gen.py \
  --flores datasets/flores_en_pt_cache.json \
  --samples 500 \
  --prompt-type low_scores \
  --output judge_training_data_low_scores.json
```

### Generate middle-range data (for ambiguous cases):
```bash
python pt_app/eval_model/judge_gen.py \
  --flores datasets/flores_en_pt_cache.json \
  --samples 300 \
  --prompt-type middle_range \
  --output judge_training_data_middle_range.json
```

### Generate balanced data (original):
```bash
python pt_app/eval_model/judge_gen.py \
  --flores datasets/flores_en_pt_cache.json \
  --samples 500 \
  --prompt-type balanced \
  --output judge_training_data_balanced.json
```

### Combine datasets:
```python
import json

# Load all datasets
with open('judge_training_data_balanced.json') as f:
    balanced = json.load(f)
    
with open('judge_training_data_low_scores.json') as f:
    low_scores = json.load(f)

with open('judge_training_data_middle_range.json') as f:
    middle_range = json.load(f)

# Combine (you can adjust weights by using different sample sizes)
combined = balanced + low_scores + middle_range

# Save
with open('judge_training_data_combined.json', 'w') as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
```

## Why Multiple Prompt Types?

Current model issues identified through prediction analysis:

**Low Scores Needed (1-5):**
- Model gives 9/10 to translations with major semantic errors
- Calls wrong words "less natural" instead of wrong
- Doesn't distinguish style vs meaning errors

**Middle Range Needed (4-8):**
- Model struggles with ambiguous quality (scores 4-7)
- Unclear boundary between "poor but acceptable" and "unacceptable"
- Strong at extremes but weak at borderline cases
- Only 61% accuracy within 1 point for middle-quality translations
- Only 40/130 predictions in "Excellent (≤1)" category

**What each prompt type teaches:**

- **Low-score data**: What "bad" looks like (scores 1-4), major semantic errors vs minor style issues, when to give harsh scores
- **Middle-range data**: Boundary between acceptable/unacceptable, gradations in quality (is it 5 or 6?), how multiple minor errors accumulate
- **Balanced data**: Full spectrum baseline, high-quality translations, subtle issues

## Recommended Strategies

### Strategy 1: Address Over-Optimism (Main Issue)
1. Generate 500 low-score examples (2000 variations)
2. Generate 200 middle-range examples (800 variations) 
3. Combine with existing balanced data (~1372 examples)
4. Re-split into train/val/test (80/10/10)
5. Retrain with rank=32, early stopping

**Total: ~4,172 examples** (was 1,372)
- Low scores: ~48% (addresses main weakness)
- Middle range: ~19% (improves boundary detection)
- Balanced: ~33% (maintains baseline)

### Strategy 2: Lighter Mix (Use if compute limited)
1. Generate 300 low-score examples (1200 variations)
2. Generate 150 middle-range examples (600 variations)
3. Keep existing balanced data
4. Retrain

**Total: ~3,172 examples**
- Better than current, less compute than Strategy 1
