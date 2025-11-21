# Judge Model Training - Configuration Guide for M1 Pro (16GB)

## System: M1 Pro with 16GB RAM

### ✅ RECOMMENDED: Llama-3.2-1B-Instruct

**Why:**
- Fits comfortably in 16GB RAM
- Fast training (~20-30 min for 3 epochs)
- Good performance for judge task

**Config:**
```bash
python pt_app/eval_model/judge_train_mps.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --epochs 3 \
    --lr 2e-4 \
    --max_seq_length 512
```

**Memory Usage:**
- Model: ~4GB
- Gradients: ~1-2GB
- Activations: ~2-3GB
- **Total: ~8-10GB** ✅ Safe

**Training Speed:**
- ~100-150 samples/min
- 3 epochs with 800 samples: ~20-30 min

---

### ⚠️ POSSIBLE: Llama-3.2-3B-Instruct

**Why:**
- Might give better performance
- Will be tight on memory
- Slower training

**Config:**
```bash
python pt_app/eval_model/judge_train_mps.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --epochs 3 \
    --lr 2e-4 \
    --max_seq_length 512
```

**Memory Usage:**
- Model: ~10-12GB
- Gradients: ~2-3GB
- Activations: ~2-3GB
- **Total: ~14-16GB** ⚠️ Tight (may OOM)

**Training Speed:**
- ~40-60 samples/min
- 3 epochs with 800 samples: ~60-90 min

**If OOM occurs:**
- Reduce `max_seq_length` to 384
- Enable gradient checkpointing (edit script, set `gradient_checkpointing=True`)
- Reduce batch size to 1 (already default for 3B)

---

### ❌ NOT RECOMMENDED: Qwen-2.5-3B

**Why:**
- Similar size to Llama-3B
- Less tested on MPS
- Same memory constraints

---

## Key Differences from Original Script

| Feature | Original (judge_train.py) | MPS Optimized (judge_train_mps.py) |
|---------|---------------------------|-------------------------------------|
| Quantization | `load_in_4bit=True` ❌ | None (float32) ✅ |
| Device | `device_map="auto"` ❌ | Manual MPS ✅ |
| Precision | `fp16=True` ❌ | float32 ✅ |
| Optimizer | `paged_adamw_8bit` ❌ | `adamw_torch` ✅ |
| Memory Clear | No | Every 5-10 steps ✅ |
| Batch Size | 2 | 2 (1B) / 1 (3B) ✅ |
| Data Loading | Standard | No multiprocessing ✅ |

---

## Quick Start

### 1. Check your data file exists:
```bash
ls -lh judge_training_data_cleaned.json
```

### 2. Start training with 1B model (RECOMMENDED):
```bash
python pt_app/eval_model/judge_train_mps.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --epochs 3 \
    --output_dir ./judge_adapters
```

### 3. Monitor memory during training:
```bash
# In another terminal
watch -n 1 'ps aux | grep python | grep judge_train'
```

---

## Troubleshooting

### If you see "MPS out of memory":
1. Kill the process (Ctrl+C)
2. Clear MPS cache:
   ```python
   import torch
   torch.mps.empty_cache()
   ```
3. Restart with smaller config:
   - Reduce `--max_seq_length 384`
   - Or switch to 1B model if using 3B

### If training is too slow:
- Increase `gradient_accumulation_steps` (edit script)
- Accept longer training time (~60-90 min for 3B)

### If you need better performance:
- Try 3B model with reduced sequence length
- Or train 1B longer (5-10 epochs)

---

## Expected Results

**With 1B model:**
- Training loss: 0.5 - 1.0
- Validation loss: 0.6 - 1.2
- Judge should give reasonable scores (1-5 scale)

**With 3B model (if it fits):**
- Training loss: 0.4 - 0.8
- Validation loss: 0.5 - 1.0
- Slightly better discrimination between good/bad translations

---

## After Training

Your adapter will be saved to `./judge_adapters/final/`

**To use it:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct"
)
model = PeftModel.from_pretrained(base_model, "./judge_adapters/final")
tokenizer = AutoTokenizer.from_pretrained("./judge_adapters/final")
```
