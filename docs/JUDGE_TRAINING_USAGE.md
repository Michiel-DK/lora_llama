# Judge Model Training - MPS Optimized

## Quick Start

Train Qwen 3B as translation quality judge on M1 Pro (16GB):

```bash
python pt_app/eval_model/judge_train_mps.py
```

Default configuration (optimized for Qwen 3B):
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Batch size**: 1 (per device)
- **Gradient accumulation**: 2 (effective batch = 2)
- **Learning rate**: 1e-4
- **Max sequence length**: 2048
- **Epochs**: 3
- **Output**: `./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-{timestamp}_final`

## Features

✅ **WandB Integration**: Automatic logging to `EN_PT_TRANSLATION_LORA` project
✅ **Weave Integration**: Experiment tracking enabled
✅ **MPS Optimized**: Cache clearing every 5 steps, float32 precision
✅ **LoRA**: Rank 8, Alpha 16 for efficient fine-tuning
✅ **Data Loading**: Automatically loads from `datasets/judge_eval/`
✅ **Naming Convention**: Follows same format as main training script

## Data Structure

The script expects these files in `datasets/judge_eval/`:
- `judge_train.json` - Training examples
- `judge_val.json` - Validation examples  
- `judge_test.json` - Test examples

Each JSON should contain list of objects with:
```json
{
  "input": "system and user prompt for judge",
  "output": "expected judge response"
}
```

## Custom Configuration

### Different Model
```bash
python pt_app/eval_model/judge_train_mps.py \
  --model meta-llama/Llama-3.2-1B-Instruct
```

### Adjust Memory Usage
```bash
# Reduce memory (if OOM)
python pt_app/eval_model/judge_train_mps.py \
  --batch_size 1 \
  --gradient_accumulation 1 \
  --max_seq_length 1024

# Increase throughput (if memory available)
python pt_app/eval_model/judge_train_mps.py \
  --batch_size 1 \
  --gradient_accumulation 4
```

### More Epochs
```bash
python pt_app/eval_model/judge_train_mps.py \
  --epochs 5
```

### Custom Project Name
```bash
python pt_app/eval_model/judge_train_mps.py \
  --project_name "MY_JUDGE_PROJECT"
```

## Memory Estimates

| Model | Batch Size | Grad Accum | Est. Memory | Status |
|-------|-----------|-----------|-------------|---------|
| Llama-3.2-1B | 2 | 2 | ~8GB | ✅ Safe |
| Llama-3.2-3B | 1 | 2 | ~12GB | ⚠️ Tight |
| Qwen-2.5-3B | 1 | 2 | ~12GB | ⚠️ Tight |

**Recommended**: Qwen 3B with batch_size=1, gradient_accumulation=2

## WandB Logging

The script automatically logs to WandB with:
- **Project**: EN_PT_TRANSLATION_LORA (or custom via `--project_name`)
- **Run name**: `Qwen2.5-3B-Instruct-judge-3ep-20250120_143022`
- **Tags**: `judge_training`, `mps`, `lora`
- **Metrics**: 
  - Training loss
  - Validation loss  
  - Learning rate
  - Gradient norm

View runs at: https://wandb.ai/your-username/EN_PT_TRANSLATION_LORA

## Output Structure

```
adapters_eval/
└── Qwen2.5-3B-Instruct-judge-3ep-20250120_143022_final/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── special_tokens_map.json
```

## Loading Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float32,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20250120_143022_final"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20250120_143022_final"
)
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce batch size to 1: `--batch_size 1`
2. Reduce gradient accumulation: `--gradient_accumulation 1`
3. Reduce max sequence length: `--max_seq_length 1024`
4. Try smaller model: `--model meta-llama/Llama-3.2-1B-Instruct`

### Slow Training
- Expected: ~30-60 min per epoch on M1 Pro with Qwen 3B
- MPS cache clearing every 5 steps prevents memory buildup
- Monitor in WandB for detailed timing

### Data Loading Errors
Ensure files exist:
```bash
ls -lh datasets/judge_eval/judge_*.json
```

Should show:
- judge_train.json
- judge_val.json
- judge_test.json

### WandB Not Logging
1. Login to WandB: `wandb login`
2. Check API key is set
3. Verify internet connection
4. Check `report_to=["wandb"]` in TrainingArguments

## Comparison with Original judge_train.py

| Feature | judge_train.py | judge_train_mps.py |
|---------|---------------|-------------------|
| Quantization | 4-bit | None (MPS incompatible) |
| Precision | fp16 | float32 |
| Optimizer | paged_adamw_8bit | adamw_torch |
| Default Model | Qwen 3B | Qwen 3B |
| Default Batch | 2 | 1 |
| Grad Accum | 4 | 2 |
| Device | CUDA | MPS |
| Cache Clearing | No | Yes (every 5 steps) |
| WandB | Manual | Automatic |
| Weave | No | Yes |

## Performance Tips

1. **Monitor memory**: Use Activity Monitor to watch RAM usage
2. **Cache clearing**: Happens automatically every 5 steps
3. **Gradient checkpointing**: Can be enabled if OOM persists
4. **Save frequency**: Model saves only at end by default
5. **Validation**: Runs every epoch automatically

## Next Steps

After training completes:
1. Check WandB dashboard for training curves
2. Load model and test on sample judge tasks
3. Compare against baseline Qwen 3B
4. Evaluate on held-out test set
5. Use in translation evaluation pipeline
