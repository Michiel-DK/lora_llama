#!/bin/bash
# setup_vast.sh - Quick setup script for Vast.ai instances

set -e  # Exit on error

echo "=========================================="
echo "Vast.ai GPU Training Setup"
echo "=========================================="

# 1. Check GPU
echo -e "\n1. Checking GPU..."
nvidia-smi || { echo "❌ No GPU found!"; exit 1; }
python3 -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"

# 2. Update system
echo -e "\n2. Updating system packages..."
apt-get update -qq
apt-get install -y git vim screen htop

# 3. Upgrade pip
echo -e "\n3. Upgrading pip..."
pip install --upgrade pip -q

# 4. Install PyTorch with CUDA
echo -e "\n4. Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# 5. Install ML packages
echo -e "\n5. Installing ML packages..."
pip install transformers datasets peft trl accelerate bitsandbytes -q

# 6. Install monitoring
echo -e "\n6. Installing monitoring tools..."
pip install wandb weave -q

# 7. Install metrics
echo -e "\n7. Installing metrics packages..."
pip install scikit-learn scipy rouge-score numpy pandas -q

# 8. Verify installation
echo -e "\n8. Verifying installation..."
python3 -c "
import torch
import transformers
import peft
import bitsandbytes
import wandb

print('✅ All packages imported successfully!')
print(f'   PyTorch: {torch.__version__}')
print(f'   Transformers: {transformers.__version__}')
print(f'   PEFT: {peft.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo -e "\n=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clone your repo: git clone <your_repo>"
echo "2. Set WandB key: export WANDB_API_KEY='your_key'"
echo "3. Login to WandB: wandb login"
echo "4. Run training in screen:"
echo "   screen -S training"
echo "   python pt_app/eval_model/judge_train_cuda.py --epochs 3"
echo "   Ctrl+A then D to detach"
echo ""
