# Quick Inference Reference

## 🚀 Fastest Way to Get Started

```bash
# Interactive mode - just start typing!
python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --interactive
```

## 📝 Three Ways to Use

### 1. Command Line (run_inference.py)

```bash
# Single prompt
python run_inference.py --adapter PATH --prompt "Your text here"

# Interactive
python run_inference.py --adapter PATH --interactive

# Batch from file
python run_inference.py --adapter PATH --input prompts.txt --output results.txt
```

### 2. Python Script

```python
from pt_app.trainer.inference import TranslationInference

translator = TranslationInference()
translator.load_adapter("./adapters/YOUR_ADAPTER", use_cpu=True)

# Single translation
result = translator.translate("Hello!")
print(result['filtered'])

# Batch translation
results = translator.batch_translate(["Hello!", "Goodbye!"])

# OR interactive mode
translator.interactive()
```

### 3. Example Scripts

```bash
python inference_examples.py
# Choose from 4 examples
```

## 🎯 Common Use Cases

### Just Testing Things Out
```bash
python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --interactive
```

### Translating One Thing
```python
from pt_app.trainer.inference import TranslationInference

translator = TranslationInference()
translator.load_adapter("./adapters/20251118_124144_best_ep1", use_cpu=True)
result = translator.translate("How are you?")
print(result['filtered'])  # Como você está?
```

### Many Translations
```bash
# Create prompts.txt with one English sentence per line
python run_inference.py \
    --adapter ./adapters/20251118_124144_best_ep1 \
    --input prompts.txt \
    --output translations.txt
```

## ⚙️ Key Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `temperature` | 0.8 | Lower (0.3) = deterministic, Higher (1.2) = creative |
| `max_new_tokens` | 150 | Maximum translation length |
| `use_quality_filter` | True | Clean up output (recommended: keep on) |
| `use_cpu` | False | Use CPU instead of GPU/MPS |

## 🔧 Troubleshooting

**Memory error on Mac?**
```python
trainer.load_adapter_for_inference(adapter_path, use_cpu=True)
```

**Translation quality issues?**
- Try temperature 0.5-0.7 (more conservative)
- Make sure quality filter is on (default)
- Use a `best_ep*` adapter, not `final`

**Can't find adapter?**
```bash
ls -lh adapters/  # List your adapters
```

## 📂 Your Adapters

Located in `./adapters/`:
- `YYYYMMDD_HHMMSS_best_ep1` ← **Use these** (best performing)
- `YYYYMMDD_HHMMSS_final` ← Final after all epochs

## 💡 Pro Tips

1. **Start with interactive mode** to test your adapter
2. **Use best_ep adapters** - they're optimized
3. **Keep quality filter on** - cleaner outputs
4. **Temperature 0.7-0.8** - good default
5. **On Mac, use `use_cpu=True`** - more stable

## 📚 More Info

- Full guide: `INFERENCE_GUIDE.md`
- Examples: `inference_examples.py`
- Help: `python run_inference.py --help`
