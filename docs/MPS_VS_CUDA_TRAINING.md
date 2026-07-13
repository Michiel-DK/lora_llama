# Training Script Comparison: MPS vs CUDA

## 📁 Files

- **`judge_train_mps.py`** - For local training on Apple Silicon (M1/M2/M3)
- **`judge_train_cuda.py`** - For cloud training on NVIDIA GPUs (Vast.ai)

---

## 🔄 Key Differences

| Feature | MPS (Local) | CUDA (Vast.ai) |
|---------|-------------|----------------|
| **File** | `judge_train_mps.py` | `judge_train_cuda.py` |
| **Device** | Apple M1/M2/M3 | NVIDIA GPU (GTX/RTX) |
| **Quantization** | None (float32) | 4-bit QLoRA (optional) |
| **Precision** | float32 | fp16 mixed precision |
| **Optimizer** | adamw_torch | paged_adamw_8bit |
| **Batch size** | 1 (limited by RAM) | 2-4 (VRAM dependent) |
| **Default model** | Qwen2-1.5B-Instruct | Qwen2.5-3B-Instruct |
| **Memory mgmt** | MPS cache clearing | CUDA auto-management |
| **Speed** | ~4-5 hours for 3 epochs | ~2-3 hours for 3 epochs |
| **Cost** | Free (your hardware) | ~$0.10-0.15 per training |
| **Dependencies** | Standard PyTorch | + bitsandbytes |

---

## 🎯 When to Use Which

### Use `judge_train_mps.py` when:
- ✅ You have Apple Silicon Mac (M1/M2/M3)
- ✅ You want free training (no cloud costs)
- ✅ You're doing experiments/testing
- ✅ Training time isn't critical
- ✅ You want to train 1B-1.5B models

### Use `judge_train_cuda.py` when:
- ✅ You need faster training (3-5x speedup)
- ✅ You want to train larger models (3B-7B)
- ✅ You have Vast.ai/cloud GPU access
- ✅ You need 4-bit quantization for memory
- ✅ Training time is important (production)

---

## 💻 Usage Examples

### MPS (Local Apple Silicon)
```bash
# Default: Qwen 1.5B, 3 epochs, batch=1
python pt_app/eval_model/judge_train_mps.py --epochs 3

# Quick test
python pt_app/eval_model/judge_train_mps.py --epochs 1 --max_samples 50

# Llama 1B variant
python pt_app/eval_model/judge_train_mps.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --epochs 2
```

### CUDA (Vast.ai / Cloud GPU)
```bash
# Default: Qwen 2.5 3B, 3 epochs, batch=2, 4-bit
python pt_app/eval_model/judge_train_cuda.py --epochs 3

# Quick test
python pt_app/eval_model/judge_train_cuda.py --epochs 1 --max_samples 50

# No quantization (needs more VRAM)
python pt_app/eval_model/judge_train_cuda.py --no_4bit

# 8GB GPU (lower batch)
python pt_app/eval_model/judge_train_cuda.py \
    --batch_size 1 \
    --gradient_accumulation 4
```

---

## 🔧 Code Changes Summary

### 1. Import Changes
```python
# MPS
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# CUDA (added)
from transformers import ..., BitsAndBytesConfig
from peft import ..., prepare_model_for_kbit_training
```

### 2. Model Loading

**MPS:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # MPS works best with float32
    trust_remote_code=True,
    cache_dir="./cache/",
    use_cache=False,
)
model = model.to(device)  # Manual device placement
```

**CUDA:**
```python
# With 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    cache_dir="./cache/",
    device_map="auto",  # Automatic device placement
    use_cache=False,
)

model = prepare_model_for_kbit_training(model)
# No manual .to(device) - device_map handles it
```

### 3. Training Arguments

**MPS:**
```python
TrainingArguments(
    # No FP16/BF16
    use_cpu=False,  # Use MPS
    dataloader_pin_memory=False,  # MPS doesn't support
    dataloader_num_workers=0,  # Avoid multiprocessing
    optim="adamw_torch",
    gradient_checkpointing=False,  # Optional
)
```

**CUDA:**
```python
TrainingArguments(
    fp16=True,  # Mixed precision
    dataloader_pin_memory=True,  # CUDA supports
    dataloader_num_workers=4,  # Parallel loading
    optim="paged_adamw_8bit",  # 8-bit optimizer
    gradient_checkpointing=True,  # Save memory
    max_grad_norm=0.3,  # More conservative
)
```

### 4. Custom Trainer

**MPS:**
```python
# Custom MPSTrainer with periodic cache clearing
class MPSTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.state.global_step % self.clear_cache_steps == 0:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        return loss
```

**CUDA:**
```python
# Standard Trainer - CUDA manages memory well
trainer = Trainer(...)
```

### 5. Device Detection

**MPS:**
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**CUDA:**
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
```

---

## 📊 Performance Comparison

### Training Speed (3 epochs on judge dataset)

| Configuration | Time | Cost |
|--------------|------|------|
| **MPS: M1 Pro 16GB, Qwen 1.5B, float32** | 4-5 hours | Free |
| **MPS: M1 Pro 16GB, Llama 1B, float32** | 3-4 hours | Free |
| **CUDA: RTX 3060 12GB, Qwen 3B, 4-bit** | 2-3 hours | $0.08-0.12 |
| **CUDA: GTX 1080 Ti 11GB, Qwen 3B, 4-bit** | 3-4 hours | $0.23-0.31 |
| **CUDA: RTX 4060 16GB, Qwen 3B, fp16** | 1.5-2 hours | $0.15-0.20 |

### Memory Usage

| Configuration | Peak Memory |
|--------------|-------------|
| **MPS: Qwen 1.5B, float32, batch=1** | ~8GB RAM |
| **MPS: Llama 1B, float32, batch=1** | ~6GB RAM |
| **CUDA: Qwen 3B, 4-bit, batch=2** | ~8GB VRAM |
| **CUDA: Qwen 3B, fp16, batch=2** | ~12GB VRAM |

---

## 🔄 Migration Guide

### From MPS to CUDA

1. **Copy your data:**
```bash
# Upload datasets to Vast.ai
scp -P <port> -r datasets/judge_eval root@<ip>:/root/lora_llama/datasets/
```

2. **Use CUDA script:**
```bash
# On Vast.ai instance
python pt_app/eval_model/judge_train_cuda.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3
```

3. **Download adapters:**
```bash
# From local machine
scp -P <port> -r root@<ip>:/root/lora_llama/adapters_eval/*_final ./
```

### From CUDA to MPS

1. **Download trained adapter** from Vast.ai

2. **Use locally:**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load on MPS/CPU
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float32,  # MPS likes float32
    device_map="mps"  # or "cpu"
)

model = PeftModel.from_pretrained(base_model, "./adapter_path")
```

---

## 🎓 Best Practices

### MPS Training
1. ✅ Use smaller models (1B-1.5B)
2. ✅ Keep max_seq_length=512
3. ✅ batch_size=1 is usually optimal
4. ✅ Monitor Activity Monitor for RAM usage
5. ✅ Use dynamic padding (already enabled)

### CUDA Training
1. ✅ Use 4-bit for 8GB GPUs
2. ✅ Disable 4-bit for 16GB+ GPUs (faster)
3. ✅ batch_size=2-4 depending on VRAM
4. ✅ Monitor with `nvidia-smi`
5. ✅ Use gradient_checkpointing for memory
6. ✅ Enable fp16 for speed
7. ✅ Run in `screen` session on Vast.ai

---

## 🚀 Quick Start Commands

### Test Locally (MPS)
```bash
# 1 epoch, 50 samples, ~10 min
python pt_app/eval_model/judge_train_mps.py \
    --epochs 1 \
    --max_samples 50 \
    --test_samples 20
```

### Full Training (CUDA)
```bash
# Full dataset, 3 epochs, ~2-3 hours
python pt_app/eval_model/judge_train_cuda.py \
    --epochs 3 \
    --batch_size 2
```

### Both Scripts Support Same Args
```bash
--model <model_name>
--epochs <int>
--batch_size <int>
--gradient_accumulation <int>
--lr <float>
--max_seq_length <int>
--max_samples <int>      # Limit training data
--test_samples <int>     # Limit test data
--project_name <string>  # WandB project
```

**CUDA-only:**
```bash
--no_4bit  # Disable 4-bit quantization
```

---

## 📝 Notes

- Both scripts produce **identical adapter formats** - interchangeable
- Both use **same data format** - `datasets/judge_eval/`
- Both support **same models** from HuggingFace
- Both log to **WandB and Weave**
- Evaluation always runs on **CPU** in both scripts (memory safety)

The main difference is **where training happens** (local vs cloud) and **how memory is optimized** (float32 vs 4-bit QLoRA).
