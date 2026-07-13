#!/bin/bash
# Script to upload model and original test set to VM, then evaluate

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found${NC}"
    exit 1
fi

source .env

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Uploading model and test set to VM${NC}"
echo -e "${BLUE}================================================${NC}"

# Create directories on VM
echo -e "${GREEN}Creating directories on VM...${NC}"
ssh -p $VM_PORT $VM_USER@$VM_HOST "mkdir -p ~/lora_llama/adapters_eval ~/lora_llama/datasets/judge_eval ~/lora_llama/pt_app/eval_model"

# Upload the new model
echo -e "${GREEN}Uploading new model (this will take a while)...${NC}"
scp -P $VM_PORT -r adapters_eval/Qwen2.5-3B-Instruct-judge-10ep-20251215_184033_final $VM_USER@$VM_HOST:~/lora_llama/adapters_eval/

# Upload original test set (judge_test.json has correct format)
echo -e "${GREEN}Uploading test set...${NC}"
scp -P $VM_PORT datasets/judge_eval/judge_test.json $VM_USER@$VM_HOST:~/lora_llama/datasets/judge_eval/

# Upload eval script and requirements
echo -e "${GREEN}Uploading eval script and requirements...${NC}"
scp -P $VM_PORT pt_app/eval_model/eval_judge_fast.py $VM_USER@$VM_HOST:~/lora_llama/pt_app/eval_model/
scp -P $VM_PORT pt_app/eval_model/judge_train_cuda.py $VM_USER@$VM_HOST:~/lora_llama/pt_app/eval_model/
scp -P $VM_PORT requirements_cuda.txt $VM_USER@$VM_HOST:~/lora_llama/

# Upload .env file
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
echo -e "  3. Install requirements: ${BLUE}pip install -r requirements_cuda.txt${NC}"
echo -e "  4. Run eval: ${BLUE}python3 pt_app/eval_model/eval_judge_fast.py --adapter adapters_eval/Qwen2.5-3B-Instruct-judge-10ep-20251215_184033_final${NC}"
echo -e "  5. Download results: ${BLUE}scp -P $VM_PORT $VM_USER@$VM_HOST:~/lora_llama/adapters_eval/Qwen2.5-3B-Instruct-judge-10ep-20251215_184033_final/test_predictions_fast.json ./original_test_predictions.json${NC}"
echo ""
