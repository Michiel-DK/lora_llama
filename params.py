# params.py (HuggingFace compatible)
import os
from datetime import datetime

HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

PROJECT_NAME = 'EN_PT_TRANSLATION_LORA'

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
NEW_MODEL_NAME = "new-model"
USER_NAME = "mlx-community"

# Paths
ADAPTER_PATH = "./adapters/"
OUTPUT_DIR = "./outputs/"
CACHE_DIR = "./cache/"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Data Configuration
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 150
MIN_WORDS = 5

MIN_TOTAL_TOKENS = 7
MAX_TOTAL_TOKENS = 512

# ============================================================================
# DATASET CONFIGURATION - ⚠️ UPDATED!
# ============================================================================
# Options: 'opus_books', 'opensubtitles', 'opus100', 'tatoeba'
DATASET = 'opensubtitles'  # ← CHANGED from 'tatoeba'!

# Sweet spot: 500 samples (gave BLEU 34 vs 1404 → BLEU 30)
# Use 500-1000 for best results
DATASET_SAMPLES = 200  # ← CHANGED from 2000!
EPOCHS = 1

# ============================================================================
# MODEL SIZE SPECIFIC SETTINGS
# ============================================================================
if "1B" in MODEL_NAME:
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 1
    LORA_RANK = 16
    LORA_DROPOUT = 0.1
    
elif "3B" in MODEL_NAME:
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LORA_RANK = 16
    LORA_DROPOUT = 0.1
    
else:  # 8B+
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LORA_RANK = 8
    LORA_DROPOUT = 0.1  # Higher for larger model (prevent overfitting)

# LoRA Configuration
LORA_CONFIG = {
    "r": LORA_RANK,
    "lora_alpha": LORA_RANK * 2,
    "lora_dropout": LORA_DROPOUT,  # ← Also dynamic!
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
    "learning_rate": 1e-4,  # ← CHANGED from 1e-4 (faster convergence)
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}

# Training Arguments (HuggingFace format)
TRAINING_ARGS = {
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "logging_steps": 50,
    "eval_steps": 100,  # ← CHANGED from 20 (less frequent)
    "save_steps": 100,  # ← CHANGED from 20 (less frequent)
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "gradient_checkpointing": True,
    "report_to": "wandb",  # ← CHANGED from "none" (enable wandb)
    "seed": 42,
}

# Quantization Configuration (BitsAndBytes format)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# ============================================================================
# GENERATION STRATEGY CONFIGURATIONS
# ============================================================================
GENERATION_CONFIGS = {
    "greedy": {
        "strategy": "greedy",
        "max_new_tokens": 150,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
    "beam_search": {  # ← BEST strategy! Use this!
        "strategy": "beam_search",
        "max_new_tokens": 150,
        "num_beams": 4,
        "early_stopping": True,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
    "sampling": {
        "strategy": "sampling",
        "max_new_tokens": 150,
        "temperature": 0.3,
        "top_p": 0.95,
        "do_sample": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
    },
}

# Default generation strategy to use
DEFAULT_GENERATION_STRATEGY = "beam_search"  # ← CHANGED from "greedy"

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.005

# WandB configuration
USE_WANDB = True
WANDB_PROJECT = PROJECT_NAME

#### JUDGE DATA FORMATTING AND SPLITTING ####
JUDGE_DATA_FILE = os.path.join(os.path.dirname(__file__),'datasets', 'judge_eval', 'judge_training_data_merged4.json')