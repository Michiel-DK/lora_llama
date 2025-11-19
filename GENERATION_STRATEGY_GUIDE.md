# Generation Strategy Testing Guide

## Overview

You can now test and compare three different text generation strategies for translation:

1. **Greedy Decoding** - Fast, deterministic, always picks most likely token
2. **Beam Search** - Explores multiple alternatives, often produces better quality
3. **Sampling** - Low-temperature sampling for controlled creativity

## Configuration

All generation strategies are configured in `params.py`:

```python
GENERATION_CONFIGS = {
    "greedy": {
        "strategy": "greedy",
        "max_new_tokens": 150,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
    "beam_search": {
        "strategy": "beam_search",
        "max_new_tokens": 150,
        "num_beams": 4,
        "early_stopping": True,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
    "sampling": {
        "strategy": "sampling",
        "max_new_tokens": 150,
        "temperature": 0.3,
        "top_p": 0.95,
        "do_sample": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
}
```

## Usage

### Option 1: Compare Baseline vs Fine-tuned (Most Important!)

**Before testing strategies**, compare your fine-tuned model against the base model to measure improvement:

```bash
# Compare base model vs fine-tuned model
python compare_baseline_vs_finetuned.py \
    --adapter ./adapters/20251118_131246_best_ep2 \
    --max-samples 50 \
    --strategy beam_search
```

This shows:
- Base model performance (no LoRA)
- Fine-tuned model performance (with adapters)
- Absolute and percentage improvements
- Comparison table in WandB

### Option 2: Compare All Strategies (Recommended)

Use the comparison script to test all strategies and log results to WandB:

```bash
# Test all strategies on your best adapter
python compare_generation_strategies.py \
    --adapter ./adapters/20251118_131246_best_ep2 \
    --max-samples 50

# Test specific strategies only
python compare_generation_strategies.py \
    --adapter ./adapters/20251118_131246_best_ep2 \
    --max-samples 50 \
    --strategies greedy beam_search

# Test on full dataset
python compare_generation_strategies.py \
    --adapter ./adapters/20251118_131246_best_ep2
```

This will:
- Run evaluation with each strategy
- Log metrics to WandB for each strategy
- Create a comparison table showing all metrics side-by-side
- Identify the best strategy for each metric

### Option 3: Test Single Strategy

Modify your training script to test a specific strategy:

```python
from pt_app.trainer.trainer_pt import UniversalTrainer

trainer = UniversalTrainer()
model, tokenizer = trainer.get_model()

# Test with greedy decoding
results = trainer.test_generation(
    adapter_path="./adapters/20251118_131246_best_ep2",
    test_dataset=test_dataset,
    generation_strategy="greedy",  # or "beam_search" or "sampling"
    use_quality_filter=True
)

print(f"BLEU: {results['metrics']['bleu']:.2f}")
```

### Option 4: Test Base Model Only

To test **just** the base model without any adapters:

```python
from pt_app.trainer.trainer_pt import UniversalTrainer

trainer = UniversalTrainer()
model, tokenizer = trainer.get_model(apply_lora=False)  # ← No LoRA!

baseline_results = trainer.test_generation(
    adapter_path=None,  # ← No adapter = base model
    test_dataset=test_dataset,
    generation_strategy="beam_search",
    use_quality_filter=True
)

print(f"Base model BLEU: {baseline_results['metrics']['bleu']:.2f}")
```

### Option 5: Interactive Testing

Modify `run_inference.py` to accept strategy parameter (future enhancement).

## What Gets Logged to WandB

For each strategy test, the following metrics are logged:

- `test/generation_strategy` - Strategy name
- `test/bleu` - BLEU score
- `test/rouge1_f1`, `test/rouge2_f1`, `test/rougeL_f1` - ROUGE scores
- `test/perplexity_mean`, `test/perplexity_median` - Perplexity metrics
- `test/filter_pass_rate` - Quality filter pass rate
- `test/avg_prediction_length` - Average translation length

When using the comparison script, additional logs include:
- `{strategy}/bleu`, `{strategy}/rouge1`, etc. - Strategy-specific metrics
- `strategy_comparison_table` - Side-by-side comparison table
- `best_strategy_bleu`, `best_strategy_rouge`, `best_strategy_perplexity` - Best performers

## What Gets Tracked in Weave

The `test_generation` method is decorated with `@weave.op()`, so Weave automatically tracks:

- Input parameters (adapter_path, generation_strategy, etc.)
- Return values (all results including metrics and examples)
- Execution time
- Full generation config used

## Modifying Strategy Parameters

To change a strategy's parameters, edit `params.py`:

```python
# Example: Make greedy more conservative
GENERATION_CONFIGS["greedy"]["repetition_penalty"] = 1.3

# Example: Increase beam search width
GENERATION_CONFIGS["beam_search"]["num_beams"] = 8

# Example: Make sampling more creative
GENERATION_CONFIGS["sampling"]["temperature"] = 0.5
```

Changes will be automatically logged to WandB in the generation config.

## Expected Results

Based on typical behavior:

- **Greedy**: Fastest, good baseline, may lack diversity
- **Beam Search**: Often best BLEU/ROUGE, slower, more consistent
- **Sampling**: Balanced quality, can be more natural

Run the comparison script to see which works best for your specific model and dataset!

## Example Output

```
================================================================================
STRATEGY COMPARISON SUMMARY
================================================================================

Metric Comparison:
--------------------------------------------------------------------------------
Strategy            BLEU    ROUGE-L  Perplexity  Pass Rate
--------------------------------------------------------------------------------
greedy             45.23     0.6234       12.45      95.2%
beam_search        47.81     0.6512       11.23      96.8%
sampling           46.15     0.6389       11.87      95.5%
--------------------------------------------------------------------------------

Best Strategies:
  BLEU: beam_search (47.81)
  ROUGE-L: beam_search (0.6512)
  Perplexity: beam_search (11.23)
================================================================================
```

## Tips

1. **Start with comparison script** - Test all three to understand trade-offs
2. **Use smaller sample size first** - Test with `--max-samples 20` for quick iteration
3. **Check WandB dashboard** - Visual comparison helps identify patterns
4. **Consider speed vs quality** - Greedy is 4x faster than beam search
5. **Filter pass rate matters** - Lower rates may indicate generation issues
