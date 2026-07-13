# Inference Guide

This guide explains how to use trained LoRA adapters for translation inference.

## Quick Start

### Option 1: Command Line (Easiest)

```bash
# Interactive mode - best for testing
python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --interactive

# Single translation
python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --prompt "Hello, how are you?"

# Batch processing from file
python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --input prompts.txt --output translations.txt
```

### Option 2: Python Script

```python
from pt_app.trainer.inference import TranslationInference

# Initialize and load adapter
translator = TranslationInference()
translator.load_adapter(
    adapter_path="./adapters/20251118_124144_best_ep1",
    use_cpu=True  # Recommended for MPS devices
)

# Translate
result = translator.translate("Hello, how are you?")
print(result['filtered'])  # 'Olá, como você está?'
```

### Option 3: Interactive Examples

```bash
# Run interactive examples
python inference_examples.py
```

## Available Adapters

Your trained adapters are in the `./adapters/` directory:

```bash
ls -lh adapters/
```

Look for directories with patterns like:
- `20251118_124144_best_ep1` - Best model from epoch 1
- `20251118_131246_best_ep2` - Best model from epoch 2
- `YYYYMMDD_HHMMSS_final` - Final model after all epochs

## Inference Methods

### 1. `load_adapter(adapter_path, use_cpu=False)`

Loads a trained adapter for inference.

**Parameters:**
- `adapter_path` (str): Path to adapter directory
- `use_cpu` (bool): Force CPU inference (recommended for MPS devices)

**Example:**
```python
translator.load_adapter(
    adapter_path="./adapters/20251118_124144_best_ep1",
    use_cpu=True
)
```

### 2. `translate(prompt, **kwargs)`

Translates a single prompt.

**Parameters:**
- `prompt` (str): English text to translate
- `max_new_tokens` (int): Maximum tokens to generate (default: 150)
- `temperature` (float): Sampling temperature, 0.0-2.0 (default: 0.8)
  - Lower (0.3-0.5): More deterministic, conservative
  - Medium (0.7-0.9): Balanced
  - Higher (1.0-1.5): More creative, variable
- `top_p` (float): Nucleus sampling parameter (default: 0.9)
- `use_quality_filter` (bool): Apply quality filtering (default: True)
- `verbose` (bool): Show filtering details (default: False)

**Returns:**
Dictionary with `'raw'` and `'filtered'` translations.

**Example:**
```python
result = translator.translate(
    prompt="Good morning!",
    temperature=0.5,  # More deterministic
    max_new_tokens=100
)
print(result['filtered'])  # Recommended output
print(result['raw'])       # Unfiltered output
```

### 3. `batch_translate(prompts, **kwargs)`

Translates multiple prompts efficiently.

**Parameters:**
- `prompts` (list): List of English texts
- `max_new_tokens` (int): Maximum tokens per translation (default: 150)
- `temperature` (float): Sampling temperature (default: 0.8)
- `use_quality_filter` (bool): Apply filtering (default: True)
- `show_progress` (bool): Show progress bar (default: True)

**Returns:**
List of translation dictionaries.

**Example:**
```python
prompts = ["Hello!", "Goodbye!", "Thank you!"]
results = translator.batch_translate(prompts)
for result in results:
    print(result['filtered'])
```

### 4. `interactive(**kwargs)`

Starts an interactive translation session.

**Parameters:**
- `max_new_tokens` (int): Maximum tokens per translation (default: 150)
- `temperature` (float): Sampling temperature (default: 0.8)
- `use_quality_filter` (bool): Apply filtering (default: True)

**Example:**
```python
translator.interactive(temperature=0.5)
# Then type prompts interactively, 'quit' to exit
```

## Command Line Options

### `run_inference.py` Arguments

**Required:**
- `--adapter PATH` - Path to adapter directory

**Mode (choose one):**
- `--interactive` - Interactive mode
- `--prompt TEXT` - Single prompt
- `--input FILE` - Batch mode input file

**Optional:**
- `--output FILE` - Output file (required with `--input`)
- `--max-tokens N` - Maximum tokens (default: 150)
- `--temperature F` - Temperature (default: 0.8)
- `--no-filter` - Disable quality filtering
- `--cpu` - Force CPU inference
- `--verbose` - Show filtering details

## Quality Filter

The quality filter post-processes translations to:
- Remove repetitions
- Fix language mixing
- Clean up artifacts
- Ensure proper sentence structure

**Recommended:** Keep enabled (default) for production use.

**Disable only for:**
- Debugging
- Comparing raw vs filtered output
- Special use cases

## Temperature Guide

Temperature controls randomness in generation:

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.1 - 0.3 | Very deterministic | Formal documents, consistency needed |
| 0.4 - 0.6 | Conservative | General translation, safe choice |
| 0.7 - 0.9 | Balanced | Default, good variety |
| 1.0 - 1.5 | Creative | Literary translation, variety needed |
| 1.5+ | Very random | Experimental only |

## Examples

### Example 1: Single Translation
```python
from pt_app.trainer.inference import TranslationInference

translator = TranslationInference()
translator.load_adapter("./adapters/20251118_124144_best_ep1", use_cpu=True)

result = translator.translate("Where is the nearest pharmacy?")
print(result['filtered'])
# Output: Onde fica a farmácia mais próxima?
```

### Example 2: Multiple Translations
```python
prompts = [
    "Good morning!",
    "Thank you very much.",
    "I don't understand."
]

for prompt in prompts:
    result = translator.translate(prompt)
    print(f"EN: {prompt}")
    print(f"PT: {result['filtered']}\n")
```

### Example 3: Batch Translation
```python
prompts = [
    "Good morning!",
    "Thank you very much.",
    "I don't understand."
]

results = translator.batch_translate(prompts, show_progress=True)
for prompt, result in zip(prompts, results):
    print(f"EN: {prompt}")
    print(f"PT: {result['filtered']}\n")
```

### Example 4: Comparing Temperatures
```python
prompt = "The cat sat on the mat."

# Deterministic
result1 = translator.translate(prompt, temperature=0.3)

# Creative
result2 = translator.translate(prompt, temperature=1.2)

print(f"T=0.3: {result1['filtered']}")
print(f"T=1.2: {result2['filtered']}")
```

### Example 5: Batch Processing
```bash
# Create input file
cat > prompts.txt << EOF
Hello, world!
How are you?
Good night.
EOF

# Process
python run_inference.py \
    --adapter ./adapters/20251118_124144_best_ep1 \
    --input prompts.txt \
    --output translations.txt \
    --temperature 0.5

# View results
cat translations.txt
```

## Troubleshooting

### Memory Issues (MPS Crashes)
**MPS (Apple Silicon) has a 4GB tensor limit that causes crashes.**

✅ **FIXED**: CPU is now the default. The inference class automatically detects MPS and uses CPU for stability.

If you still see crashes:
```
Error: total bytes of NDArray > 2**32
```

See detailed explanation in `MPS_INFERENCE_ISSUES.md`.

### Slow Inference
- **CPU (default)**: Stable, 1-2s per prompt on Mac
- **CUDA (NVIDIA)**: Fastest, 0.3-0.5s per prompt
- **MPS (Mac)**: Fast but crashes on large models - **not recommended**
- First inference is always slower (model loading ~10-15s)

### Poor Translation Quality
1. Try different temperature values (0.3-0.9)
2. Ensure quality filter is enabled
3. Check if you're using the best epoch adapter
4. Verify adapter path is correct

### Model Not Found
```bash
# List available adapters
ls -lh adapters/

# Use full/correct path
python run_inference.py --adapter ./adapters/CORRECT_NAME_HERE --interactive
```

## Best Practices

1. **Use best epoch adapters** - Look for `best_ep*` in adapter names
2. **Keep quality filter on** - Better production outputs
3. **Start with temperature 0.7-0.8** - Good balance
4. **Use CPU for MPS devices** - More stable
5. **Test interactively first** - Before batch processing

## Advanced Usage

### Custom Generation Config
```python
result = translator.translate(
    prompt="Complex sentence here",
    max_new_tokens=200,      # Longer translations
    temperature=0.5,         # More deterministic
    top_p=0.85,             # Narrower sampling
    use_quality_filter=True,
    verbose=True            # See what filter does
)
```

### Integration in Your Code
```python
from pt_app.trainer.inference import TranslationInference

class MyTranslationService:
    def __init__(self, adapter_path):
        self.translator = TranslationInference()
        self.translator.load_adapter(adapter_path, use_cpu=True)
    
    def translate(self, text):
        result = self.translator.translate(text, temperature=0.7)
        return result['filtered']

# Use it
service = MyTranslationService("./adapters/20251118_124144_best_ep1")
translation = service.translate("Hello!")
```

## Performance Tips

- **First translation is slow** (model loading) - subsequent calls are faster
- **CPU inference** is more stable on Mac
- **Batch processing** is most efficient for many translations
- **Quality filter** adds minimal overhead (~5-10ms)

## Need Help?

Check the example scripts:
```bash
# Interactive examples
python inference_examples.py

# CLI help
python run_inference.py --help
```
