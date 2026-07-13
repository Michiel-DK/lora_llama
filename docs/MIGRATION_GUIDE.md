# Migration Guide: trainer_pt.py → inference.py

Quick reference for updating your code to use the new inference class.

## Import Changes

```python
# OLD
from pt_app.trainer.trainer_pt import UniversalTrainer

# NEW
from pt_app.trainer.inference import TranslationInference
```

## Method Name Changes

| Old Method | New Method | Notes |
|------------|------------|-------|
| `UniversalTrainer()` | `TranslationInference()` | Class name change |
| `load_adapter_for_inference()` | `load_adapter()` | Shorter, clearer |
| `run_inference()` | `translate()` | More intuitive |
| `interactive_inference()` | `interactive()` | Simpler name |
| N/A | `batch_translate()` | NEW: efficient batch processing |

## Code Migration Examples

### Example 1: Simple Script

**Before:**
```python
from pt_app.trainer.trainer_pt import UniversalTrainer

trainer = UniversalTrainer()
trainer.load_adapter_for_inference("./adapters/best_ep1", use_cpu=True)
result = trainer.run_inference("Hello, world!")
print(result['filtered'])
```

**After:**
```python
from pt_app.trainer.inference import TranslationInference

translator = TranslationInference()
translator.load_adapter("./adapters/best_ep1", use_cpu=True)
result = translator.translate("Hello, world!")
print(result['filtered'])
```

### Example 2: Interactive Mode

**Before:**
```python
trainer = UniversalTrainer()
trainer.load_adapter_for_inference("./adapters/best_ep1", use_cpu=True)
trainer.interactive_inference(temperature=0.7)
```

**After:**
```python
translator = TranslationInference()
translator.load_adapter("./adapters/best_ep1", use_cpu=True)
translator.interactive(temperature=0.7)
```

### Example 3: Multiple Translations (NEW - more efficient!)

**Before:**
```python
prompts = ["Hello!", "Goodbye!", "Thank you!"]
results = []
for prompt in prompts:
    result = trainer.run_inference(prompt)
    results.append(result['filtered'])
```

**After (Better!):**
```python
prompts = ["Hello!", "Goodbye!", "Thank you!"]
results = translator.batch_translate(prompts, show_progress=True)
translations = [r['filtered'] for r in results]
```

### Example 4: Service Class

**Before:**
```python
class TranslationService:
    def __init__(self, adapter_path):
        self.trainer = UniversalTrainer()
        self.trainer.load_adapter_for_inference(adapter_path, use_cpu=True)
    
    def translate(self, text):
        result = self.trainer.run_inference(text)
        return result['filtered']
```

**After:**
```python
class TranslationService:
    def __init__(self, adapter_path):
        self.translator = TranslationInference()
        self.translator.load_adapter(adapter_path, use_cpu=True)
    
    def translate(self, text):
        result = self.translator.translate(text)
        return result['filtered']
```

## What Stayed the Same

✅ All parameters remain the same:
- `adapter_path`, `use_cpu`
- `temperature`, `max_new_tokens`, `top_p`
- `use_quality_filter`, `verbose`

✅ Return format unchanged:
- Still returns `{'raw': '...', 'filtered': '...'}`

✅ Quality filtering works exactly the same

✅ Command-line tool (`run_inference.py`) usage unchanged

## Quick Search & Replace

If you have many files to update, use these patterns:

```bash
# Find all files that import UniversalTrainer for inference
grep -r "from pt_app.trainer.trainer_pt import UniversalTrainer" .

# Replace imports (manual verification recommended)
# OLD: from pt_app.trainer.trainer_pt import UniversalTrainer
# NEW: from pt_app.trainer.inference import TranslationInference

# Replace class instantiation
# OLD: trainer = UniversalTrainer()
# NEW: translator = TranslationInference()

# Replace method calls
# OLD: .load_adapter_for_inference(
# NEW: .load_adapter(

# OLD: .run_inference(
# NEW: .translate(

# OLD: .interactive_inference(
# NEW: .interactive(
```

## Still Using Training?

If you're doing **both** training and inference, you can use both classes:

```python
from pt_app.trainer.trainer_pt import UniversalTrainer
from pt_app.trainer.inference import TranslationInference

# Training phase
trainer = UniversalTrainer()
model, tokenizer = trainer.get_model()
adapter_path = trainer.train(train_dataset, val_dataset)

# Evaluation phase (still use trainer)
results = trainer.test_generation(adapter_path, test_dataset)

# Inference phase (use dedicated inference class)
translator = TranslationInference()
translator.load_adapter(adapter_path, use_cpu=True)
translation = translator.translate("Hello!")
```

## Need Help?

- See `INFERENCE_GUIDE.md` for full documentation
- See `QUICK_START_INFERENCE.md` for quick examples
- See `inference_examples.py` for working code examples
- See `REFACTORING_SUMMARY.md` for technical details
