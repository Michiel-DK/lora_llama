# Vast.ai Training Setup Guide

## 🎯 Recommended GPU: RTX 3060 (12GB) @ $0.041/hr

**Total training cost: ~$0.10-0.15** for 3 epochs of Qwen 2.5 3B

---

## 📋 Pre-Launch Checklist

### 1. Test Locally First
```bash
# Test with small subset to verify code works
python pt_app/eval_model/judge_train_cuda.py \
    --epochs 1 \
    --max_samples 50 \
    --test_samples 20
```

### 2. Prepare WandB API Key
```bash
# Get your WandB API key from: https://wandb.ai/authorize
export WANDB_API_KEY="your_key_here"
```

---

## 🚀 Vast.ai Setup Steps

### Step 1: Rent GPU Instance

1. Go to [vast.ai](https://vast.ai)
2. Search filters:
   - **GPU**: RTX 3060, GTX 1080 Ti, or RTX 3060
   - **VRAM**: ≥8GB
   - **Disk**: ≥50GB
   - **Reliability**: >95%
   - **Max duration**: 5 hours

3. Click **"Rent"** on best option

### Step 2: SSH Into Instance

Vast.ai will provide SSH command:
```bash
ssh root@<ip_address> -p <port_number>
```

### Step 3: Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y git vim screen

# Install Python packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft trl accelerate bitsandbytes
pip install wandb weave scikit-learn scipy rouge-score

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Upload Your Code

**Option A: Git Clone (Recommended)**
```bash
# If your repo is public
git clone https://github.com/Michiel-DK/lora_llama.git
cd lora_llama

# If private, use SSH key or personal access token
```

**Option B: SCP Upload**
```bash
# From your local machine
scp -P <port> -r datasets root@<ip>:/root/lora_llama/
scp -P <port> pt_app/eval_model/judge_train_cuda.py root@<ip>:/root/lora_llama/pt_app/eval_model/
```

### Step 5: Configure WandB

```bash
# On Vast.ai instance
export WANDB_API_KEY="your_wandb_key_here"
wandb login
```

### Step 6: Run Training in Screen

```bash
# Start screen session (survives SSH disconnection)
screen -S training

# Run training
python pt_app/eval_model/judge_train_cuda.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 2 \
    --lr 2e-4

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r training
```

### Step 7: Monitor Progress

**Option 1: WandB Dashboard**
- Go to https://wandb.ai/your_username/EN_PT_TRANSLATION_LORA
- Watch real-time metrics

**Option 2: Terminal Logs**
```bash
# Reattach to screen session
screen -r training

# Or tail log file
tail -f nohup.out
```

**Option 3: GPU Monitoring**
```bash
watch -n 1 nvidia-smi
```

### Step 8: Download Trained Model

```bash
# From your local machine (after training completes)
scp -P <port> -r root@<ip>:/root/lora_llama/adapters_eval/Qwen2.5-3B-Instruct-judge-*_final ./trained_models/
```

---

## 💡 Training Configuration

### Default (Recommended for RTX 3060 12GB)
```bash
python pt_app/eval_model/judge_train_cuda.py \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 2
```

### For GTX 1080 (8GB)
```bash
python pt_app/eval_model/judge_train_cuda.py \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 4
```

### Quick Test Run
```bash
python pt_app/eval_model/judge_train_cuda.py \
    --epochs 1 \
    --max_samples 100 \
    --test_samples 20
```

---

## 📊 Expected Results

### Training Time
| GPU | Training Time | Total Cost |
|-----|--------------|------------|
| RTX 3060 12GB | 2-3 hours | $0.08-0.12 |
| GTX 1080 Ti 11GB | 3-4 hours | $0.23-0.31 |
| GTX 1080 8GB | 4-5 hours | $0.21-0.26 |

### VRAM Usage (4-bit QLoRA)
- Model: ~2GB
- Gradients: ~3GB
- Activations: ~2GB
- Buffer: ~1GB
- **Total: ~8GB** (fits in 8GB GPU!)

### Output
- **Adapter files**: `adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-TIMESTAMP_final/`
- **WandB logs**: Full training curves + test metrics
- **Test metrics**: MAE, RMSE, Cohen's Kappa, Pearson r, ROUGE-L

---

## 🔧 Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
python pt_app/eval_model/judge_train_cuda.py --batch_size 1 --gradient_accumulation 4

# Or disable 4-bit (not recommended for 8GB)
python pt_app/eval_model/judge_train_cuda.py --no_4bit
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Should see ~90-100% GPU usage
# If low, increase batch size
```

### Connection Lost
```bash
# Reattach to screen session
screen -r training

# If screen not found, training crashed - check logs
tail -100 nohup.out
```

### WandB Not Logging
```bash
# Check API key
echo $WANDB_API_KEY

# Re-login
wandb login --relogin
```

---

## 📦 Post-Training

### Download Only Adapter (Small)
```bash
# Adapters are ~50MB, base model is 3GB
scp -P <port> -r root@<ip>:/root/lora_llama/adapters_eval/Qwen2.5-3B-Instruct-judge-*_final ./
```

### Load Model Locally
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./trained_models/Qwen2.5-3B-Instruct-judge-3ep-20241208_120000_final"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Use for inference
inputs = tokenizer("User: Evaluate this translation...\n\nAssistant:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💰 Cost Optimization

### Tips to Minimize Cost
1. ✅ **Test locally first** - Don't debug on paid GPUs
2. ✅ **Use screen** - Prevents crashes from SSH disconnects
3. ✅ **Set max duration** - Prevents runaway costs (set to 4-5 hours)
4. ✅ **Download immediately** - Instance may terminate after max duration
5. ✅ **Use spot instances** - Cheaper but can be interrupted
6. ✅ **Monitor WandB** - Spot issues early without SSH

### If You Need to Stop Early
```bash
# Ctrl+C in training screen
# Then download current best checkpoint
ls -lt adapters_eval/  # Find latest checkpoint
```

---

## 🎓 Key Differences from MPS Training

| Aspect | MPS (Local) | CUDA (Vast.ai) |
|--------|-------------|----------------|
| **Quantization** | float32 | 4-bit QLoRA |
| **Precision** | float32 | fp16 |
| **Optimizer** | adamw_torch | paged_adamw_8bit |
| **Batch size** | 1 | 2 |
| **Speed** | Slower | 3-5x faster |
| **Memory** | 16GB RAM | 8-12GB VRAM |
| **Cost** | Free | $0.10/training |

---

## ✅ Success Checklist

- [ ] Local test run completed successfully
- [ ] WandB API key configured
- [ ] Vast.ai instance rented (RTX 3060 recommended)
- [ ] SSH connection established
- [ ] Dependencies installed
- [ ] Code uploaded (git or scp)
- [ ] Training started in screen session
- [ ] WandB dashboard showing live metrics
- [ ] Training completed (check screen session)
- [ ] Adapters downloaded to local machine
- [ ] Vast.ai instance destroyed (to stop billing)

---

## 🚨 Don't Forget!

**DESTROY THE INSTANCE AFTER DOWNLOADING** to stop billing:
1. Go to vast.ai dashboard
2. Click "Destroy" on your instance
3. Confirm you've downloaded your trained model first!
