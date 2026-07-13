# Continue Training from Checkpoint - Workflow

This guide shows how to upload your base adapter and training data to Vast.ai, then continue training.

## Step 1: Create Vast.ai Instance

1. Go to https://vast.ai/console/instances/
2. Search for GPU (RTX 3060 12GB or similar)
3. Select "pytorch/pytorch:latest" or CUDA-enabled image
4. Create instance and note the connection details

## Step 2: Update .env File

After creating the instance, update your `.env` file with connection details:

```bash
VM_HOST=<instance_ip>      # e.g., 1.208.108.242
VM_PORT=<ssh_port>         # e.g., 33590
VM_USER=root
```

## Step 3: Upload Files to VM

The upload script will transfer:
- Base adapter (from BEST_GEN env variable)
- Training data (judge_training_data_cleaned_1512.json)
- Training scripts
- Requirements

```bash
# Make sure .env is updated with VM connection details
./upload_to_vm.sh
```

This uploads:
- `adapters/${BEST_GEN}` → `~/lora_llama/adapters/`
- `judge_training_data_cleaned_1512.json` → `~/lora_llama/datasets/judge_eval/`
- Training scripts and dependencies

## Step 4: SSH to VM and Setup

```bash
# SSH to VM (use details from .env)
ssh -p ${VM_PORT} root@${VM_HOST}

# Navigate to project
cd ~/lora_llama

# Install dependencies
pip install -r requirements_cuda.txt

# Split training data into train/val/test
python3 split_judge_data.py datasets/judge_eval/judge_training_data_cleaned_1512.json
```

## Step 5: Start Training from Checkpoint

```bash
# Load .env variables
source .env

# Continue training from base adapter
python3 pt_app/eval_model/judge_train_cuda.py \
  --base_adapter adapters/${BEST_GEN} \
  --epochs 5 \
  --batch_size 2 \
  --gradient_accumulation 2 \
  --lr 2e-4

# Or train from scratch (no base adapter)
python3 pt_app/eval_model/judge_train_cuda.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 5
```

## What Happens

When using `--base_adapter`:
1. Base model is loaded
2. Existing LoRA adapter is loaded on top
3. Training continues from where the adapter left off
4. New adapter weights are saved

This is useful for:
- Domain adaptation: Train general model → fine-tune for specific task
- Incremental learning: Add new training data without starting over
- Transfer learning: Use adapter trained on one task as starting point for another

## Training Progress

Training will:
- Save checkpoints every N steps
- Log to WandB (using API key from .env)
- Save best and final adapters
- Run evaluation on test set

Monitor with:
```bash
# Watch training progress
tail -f nohup.out

# Or use screen/tmux for persistent session
screen -S training
python3 pt_app/eval_model/judge_train_cuda.py --base_adapter adapters/${BEST_GEN} --epochs 5
# Press Ctrl+A then D to detach
# Reconnect with: screen -r training
```

## Download Results

After training completes:

```bash
# On your local machine
./download_adapter_from_vm.sh

# Or manually with scp
scp -P ${VM_PORT} -r root@${VM_HOST}:~/lora_llama/adapters_eval/<adapter_name> ./adapters_eval/
```

## Notes

- **Base adapter model must match**: If base adapter is Llama-3.2-1B, you must use same base model
- **LoRA rank compatibility**: New training uses rank=32, old adapter may have different rank (should work but may not be optimal)
- **Memory**: QLoRA 4-bit uses ~8GB VRAM, can train 3B models on RTX 3060 12GB
- **Data format**: Script expects train/val/test JSON files in correct format
