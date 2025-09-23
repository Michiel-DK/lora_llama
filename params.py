import os

HF_TOKEN = os.getenv('HUGGINFACE_HUB_TOKEN') # <-- Add you HF Token here

MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
NEW_MODEL_NAME = "new-model"
USER_NAME = "mlx-community"

ADAPTER_PATH = "./adapters/"
MAX_SEQ_LENGTH = 512

DATASET_SAMPLES = 500

BATCH_SIZE = 4
EPOCHS = 1

LORA_CONFIG = {
    "rank": 8,
    "dropout": 0.0,
    "scale": 10.0,
    "use_dora": False,
    "num_layers": 12
}