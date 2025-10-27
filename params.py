import os

HF_TOKEN = os.getenv('HUGGINFACE_HUB_TOKEN') 

MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
NEW_MODEL_NAME = "new-model"
USER_NAME = "mlx-community"

ADAPTER_PATH = "./adapters/"
MAX_SEQ_LENGTH = 512

DATASET_SAMPLES = 50

BATCH_SIZE = 4
EPOCHS = 3

LORA_CONFIG = {
    "rank": 8,    #rank dimension for LoRA update matrices
    "dropout": 0.1,    #dropout probablitiy lora layers
    "scale": 10.0,     #magnitude of LoRA adaptations - alpha = rank * scale
    "use_dora": False,    # decomposes weights into magnitude and direction
    "num_layers": 12,
    "alpha": 32, #LoRA attention scaling parameter
    "target_modules": [    # Which modules to apply LoRA to
        "q_proj",         # Query projection
        "k_proj",         # Key projection  
        "v_proj",         # Value projection
        "o_proj",         # Output projection
        # "gate_proj",    # For MLP layers (optional)
        # "up_proj",      # For MLP layers (optional)
        # "down_proj"     # For MLP layers (optional)
    ],
    
    "bias": "none",       # How to handle biases: "none", "all", or "lora_only"
    
    # Training-specific parameters
    "task_type": "CAUSAL_LM",  # Task type for the model
    "inference_mode": False,    # Set to True during inference
}

TRAINING_CONFIG = {
    # Optimizer settings
    "learning_rate": 1e-4,      # Typical range: 1e-5 to 5e-4
    "warmup_ratio": 0.1,        # Warmup steps as ratio of total steps
    "weight_decay": 0.01,       # L2 regularization
    "adam_beta1": 0.9,          # Adam beta1 parameter
    "adam_beta2": 0.999,        # Adam beta2 parameter
    "adam_epsilon": 1e-8,       # Adam epsilon for numerical stability
    
    # Training dynamics
    "batch_size": 4,            # Per-device batch size
    "gradient_accumulation_steps": 4,  # Effective batch = batch_size * this
    "num_epochs": 3,            # Number of training epochs
    "max_steps": -1,            # Override num_epochs if set > 0
    
    # Gradient control
    "gradient_checkpointing": True,  # Trade compute for memory
    "max_grad_norm": 1.0,       # Gradient clipping threshold
    
    # MLX specific
    "use_cpu": False,           # Use CPU instead of GPU/Metal
    "seed": 42,                 # Random seed for reproducibility
}

TRAINING_ARGS = {
    "batch_size": BATCH_SIZE,
    "max_seq_length": MAX_SEQ_LENGTH,
    "grad_checkpoint": True,        # Gradient checkpointing for memory efficiency
    "steps_per_report": 20,         # Log metrics every N steps
    "steps_per_eval": 10,           # Evaluate every N steps
    "steps_per_save": 50,           # Save checkpoint every N steps
    "val_batches": 1,               # Number of validation batches to run
    "warmup_steps": 100,            # Learning rate warmup steps
    "grad_clip": 1.0,               # Gradient clipping threshold
    "loss_scale": "dynamic",        # Loss scaling for mixed precision
}


# LoRA Rank Selection Guidelines
RANK_GUIDELINES = {
    "small_dataset": 4,         # < 1K examples
    "medium_dataset": 8,        # 1K - 10K examples  
    "large_dataset": 16,        # 10K - 100K examples
    "very_large_dataset": 32,   # > 100K examples
}

# Quantization Configuration (for loading base model)
QUANTIZATION_CONFIG = {
    "bits": 4,              # 4-bit quantization
    "group_size": 64        # Quantization group size
}

# Optimizer Configuration
OPTIMIZER_CONFIG = {
    "learning_rate": 1e-5,  # Learning rate for AdamW
    "weight_decay": 0.01,   # Weight decay (L2 regularization)
    "beta1": 0.9,           # Adam beta1
    "beta2": 0.999,         # Adam beta2
    "eps": 1e-8,            # Adam epsilon
}