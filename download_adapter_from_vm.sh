#!/bin/bash
# Download trained adapter from Vast.ai VM to local machine

VM_HOST="root@1.208.108.242"
VM_PORT="33590"
ADAPTER_NAME="Qwen2.5-3B-Instruct-judge-3ep-20251209_150013_final"

# Remote path on VM
REMOTE_PATH="/root/lora_llama/adapters_eval/${ADAPTER_NAME}"

# Local path
LOCAL_PATH="./adapters_eval/${ADAPTER_NAME}"

echo "Downloading adapter from VM..."
echo "Remote: ${REMOTE_PATH}"
echo "Local: ${LOCAL_PATH}"
echo ""

# Create local directory
mkdir -p "./adapters_eval"

# Download adapter using scp with recursion
scp -P ${VM_PORT} -r ${VM_HOST}:${REMOTE_PATH} ./adapters_eval/

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Adapter downloaded successfully!"
    echo "Location: ${LOCAL_PATH}"
    echo ""
    echo "Files downloaded:"
    ls -lh "${LOCAL_PATH}"
else
    echo ""
    echo "❌ Download failed!"
    exit 1
fi
