# params.py (HuggingFace compatible)
import os

HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')

# Model Configuration
MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
NEW_MODEL_NAME = "new-model"
USER_NAME = "mlx-community"

# Paths
ADAPTER_PATH = "./adapters/"
OUTPUT_DIR = "./outputs/"
CACHE_DIR = "./cache/"

# Data Configuration
MAX_SEQ_LENGTH = 512
DATASET_SAMPLES = 200

# Training Configuration
BATCH_SIZE = 4
EPOCHS = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16

# LoRA Configuration (HuggingFace PEFT format)
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "use_dora": False,
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}

# Training Arguments (HuggingFace format - ONLY valid parameters)
TRAINING_ARGS = {
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "logging_steps": 50,
    "eval_steps": 20,
    "save_steps": 20,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "gradient_checkpointing": True,
    "report_to": "none",
    "seed": 42,
}

# Quantization Configuration (BitsAndBytes format)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}