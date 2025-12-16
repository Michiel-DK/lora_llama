#!/bin/bash
# Upload adapter and training data to Vast.ai VM

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found${NC}"
    exit 1
fi

# Load environment variables
source .env

# VM connection details (you'll update these after creating the instance)
VM_HOST="${VM_HOST:-REPLACE_WITH_VM_IP}"
VM_PORT="${VM_PORT:-REPLACE_WITH_VM_PORT}"
VM_USER="${VM_USER:-root}"

# Check if VM details are set
if [ "$VM_HOST" = "REPLACE_WITH_VM_IP" ]; then
    echo -e "${RED}ERROR: Please update VM_HOST in .env or set environment variable${NC}"
    echo "Example: export VM_HOST=1.208.108.242"
    exit 1
fi

if [ "$VM_PORT" = "REPLACE_WITH_VM_PORT" ]; then
    echo -e "${RED}ERROR: Please update VM_PORT in .env or set environment variable${NC}"
    echo "Example: export VM_PORT=33590"
    exit 1
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Uploading to Vast.ai VM${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Host: ${VM_HOST}:${VM_PORT}"
echo -e "User: ${VM_USER}"
echo ""

# Create remote directories
echo -e "${GREEN}Creating remote directories...${NC}"
ssh -p $VM_PORT $VM_USER@$VM_HOST "mkdir -p ~/lora_llama/adapters ~/lora_llama/datasets/judge_eval ~/lora_llama/pt_app/eval_model"

# Upload the base adapter (for resuming training)
if [ ! -z "$BEST_JUDGE" ]; then
    ADAPTER_PATH="./adapters_eval/${BEST_JUDGE}"
    if [ -d "$ADAPTER_PATH" ]; then
        echo -e "${GREEN}Uploading adapter: ${BEST_JUDGE}${NC}"
        scp -P $VM_PORT -r "$ADAPTER_PATH" $VM_USER@$VM_HOST:~/lora_llama/adapters/
        echo -e "${GREEN}✓ Adapter uploaded${NC}"
    else
        echo -e "${RED}WARNING: Adapter not found at ${ADAPTER_PATH}${NC}"
    fi
else
    echo -e "${RED}WARNING: BEST_JUDGE not set in .env${NC}"
fi

# Upload training data
echo -e "${GREEN}Uploading training data...${NC}"
if [ -f "datasets/judge_eval/judge_training_data_merged_1512.json" ]; then
    scp -P $VM_PORT datasets/judge_eval/judge_training_data_merged_1512.json $VM_USER@$VM_HOST:~/lora_llama/datasets/judge_eval/
    echo -e "${GREEN}✓ Training data uploaded (1121 examples)${NC}"
else
    echo -e "${RED}ERROR: datasets/judge_eval/judge_training_data_merged_1512.json not found${NC}"
    exit 1
fi

# Upload split script (to create train/val/test splits)
echo -e "${GREEN}Uploading split script...${NC}"
scp -P $VM_PORT split_judge_data.py $VM_USER@$VM_HOST:~/lora_llama/

# Upload training script
echo -e "${GREEN}Uploading training script...${NC}"
scp -P $VM_PORT pt_app/eval_model/judge_train_cuda.py $VM_USER@$VM_HOST:~/lora_llama/pt_app/eval_model/

# Upload requirements
echo -e "${GREEN}Uploading requirements...${NC}"
scp -P $VM_PORT requirements_cuda.txt $VM_USER@$VM_HOST:~/lora_llama/

# Upload .env file (contains API keys)
echo -e "${GREEN}Uploading .env file...${NC}"
scp -P $VM_PORT .env $VM_USER@$VM_HOST:~/lora_llama/

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}✓ Upload complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "Next steps on VM:"
echo -e "  1. SSH to VM: ${BLUE}ssh -p $VM_PORT $VM_USER@$VM_HOST${NC}"
echo -e "  2. Navigate: ${BLUE}cd ~/lora_llama${NC}"
echo -e "  3. Split data: ${BLUE}python3 split_judge_data.py datasets/judge_eval/judge_training_data_merged_1512.json${NC}"
echo -e "  4. Train from scratch: ${BLUE}python3 pt_app/eval_model/judge_train_cuda.py --epochs 3 --batch_size 2 --skip_eval${NC}"
echo -e "  5. Eval after training: ${BLUE}python3 pt_app/eval_model/eval_judge_fast.py --adapter adapters_eval/<adapter_name>${NC}"
echo ""
