# train_judge_mps.py - Optimized for Apple Silicon M1 Pro (16GB)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import os
from datetime import datetime
import wandb
import weave

def load_judge_datasets():
    """Load formatted train/val/test datasets from judge_eval directory"""
    
    # Load from JSON files in datasets/judge_eval
    with open("datasets/judge_eval/judge_train.json", "r") as f:
        train_data = json.load(f)
    
    with open("datasets/judge_eval/judge_val.json", "r") as f:
        val_data = json.load(f)
    
    with open("datasets/judge_eval/judge_test.json", "r") as f:
        test_data = json.load(f)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    print(f"✅ Loaded datasets:")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")
    print(f"   Test: {len(test_dataset)} examples")
    
    return train_dataset, val_dataset, test_dataset

def formatting_func(example):
    """Format examples for training - converts chat messages to text"""
    # Convert messages format to single text string
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}"
    return {"text": text}

def train_judge_mps(
    model_name="Qwen/Qwen2-1.5B-Instruct",  # Qwen2 1.5B Instruct for judge training
    output_dir="./adapters_eval",
    num_epochs=3,
    batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch size = 2
    learning_rate=1e-4,
    max_seq_length=512,  # Reduced to 512 to fit in 16GB MPS memory
    max_samples=None,  # Limit dataset size for testing
    project_name="EN_PT_TRANSLATION_LORA",
):
    """
    Fine-tune model as translation judge - OPTIMIZED FOR MPS (Apple Silicon)
    
    Default config for Qwen2 1.5B Instruct on M1 Pro 16GB:
    - batch_size=1, gradient_accumulation_steps=2 (effective batch=2)
    - max_seq_length=512 (reduced from 2048 to fit in 16GB MPS)
    - float32 precision, adamw_torch optimizer
    - WandB + Weave logging enabled
    """
    
    # Initialize WandB and Weave
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]  # "Qwen2.5-3B-Instruct"
    run_name = f"{model_short}-judge-{num_epochs}ep-{timestamp}"
    
    print("="*80)
    print(f"TRAINING JUDGE MODEL ON MPS (Apple Silicon)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Run name: {run_name}")
    print(f"Available RAM: 16GB")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print("="*80)
    
    # Initialize logging
    print("\nInitializing WandB and Weave...")
    wandb.init(
        project=project_name,
        name=run_name,
        tags=["judge_training", "mps", "lora"],
        config={
            "model": model_name,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "device": "mps",
        }
    )
    
    weave.init(project_name)
    print(f"✅ Logging to WandB: {project_name}/{run_name}")
    print(f"✅ Weave initialized: {project_name}")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("⚠️  MPS not available, using CPU (will be slower)")
    
    # Load model - NO QUANTIZATION on MPS
    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # MPS works best with float32
        trust_remote_code=True,
        cache_dir="./cache/",
        use_cache=False,  # Disable for training
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./cache/",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA configuration - OPTIMIZED FOR 16GB
    print("\n2. Applying LoRA adapters...")
    
    # Adjust LoRA rank based on model size
    if "1B" in model_name or "1.5B" in model_name or "1.8B" in model_name:
        lora_r = 16  # Higher rank for smaller models (1B-2B)
        lora_alpha = 32
    elif "3B" in model_name or "4B" in model_name:
        lora_r = 8  # Smaller rank for larger models (3B-4B)
        lora_alpha = 16
    elif "7B" in model_name:
        lora_r = 8  # Even smaller for 7B
        lora_alpha = 16
    else:
        lora_r = 8  # Default conservative
        lora_alpha = 16
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # Load datasets
    print("\n3. Loading training data...")
    train_dataset, val_dataset, test_dataset = load_judge_datasets()
    
    # Limit dataset size if max_samples specified
    if max_samples:
        print(f"   Limiting to {max_samples} samples for testing...")
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples // 5, len(val_dataset))))  # 20% for val
        print(f"   Limited train: {len(train_dataset)} examples")
        print(f"   Limited val: {len(val_dataset)} examples")
    
    # Format datasets
    train_dataset = train_dataset.map(formatting_func)
    val_dataset = val_dataset.map(formatting_func)
    test_dataset = test_dataset.map(formatting_func)
    
    # Use provided batch size and gradient accumulation
    per_device_batch = batch_size
    grad_accum = gradient_accumulation_steps
    effective_batch_size = per_device_batch * grad_accum
    clear_cache_steps = 5  # Clear MPS cache every 5 steps for stability
    
    print(f"\n4. Training configuration:")
    print(f"   Per-device batch size: {per_device_batch}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Clear cache every: {clear_cache_steps} steps")
    
    # Training arguments - MPS OPTIMIZED with WandB integration
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,  # WandB run name
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        report_to=["wandb"],  # Enable WandB logging
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # MPS-specific settings
        use_cpu=False,  # Use MPS
        dataloader_pin_memory=False,  # Don't pin memory on MPS
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        
        # NO FP16/BF16 quantization - use default float32
        # NO 8-bit optimizer - use standard AdamW
        optim="adamw_torch",
        
        # Memory management
        gradient_checkpointing=False,  # Can enable if OOM
        max_grad_norm=1.0,
    )
    
    # Custom Trainer with MPS memory management
    class MPSTrainer(Trainer):
        def __init__(self, *args, clear_cache_steps=5, **kwargs):
            super().__init__(*args, **kwargs)
            self.clear_cache_steps = clear_cache_steps
            
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Clear MPS cache periodically
            if self.state.global_step % self.clear_cache_steps == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
            
            return loss
    
    # Tokenization function - use dynamic padding for memory efficiency
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Dynamic padding handled by data collator
        )
    
    # Tokenize datasets
    print("\n5. Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Data collator for dynamic padding
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    print("\n6. Setting up trainer...")
    trainer = MPSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Add data collator for dynamic padding
        clear_cache_steps=clear_cache_steps,
    )
    
    # Train
    print("\n7. Starting training...")
    print("="*80)
    print("💡 TIP: If you see OOM errors, reduce batch_size or enable gradient_checkpointing")
    print("="*80)
    
    trainer.train()
    
    # Save final model with timestamp naming
    print("\n8. Saving model...")
    final_path = f"{output_dir}/{run_name}_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {final_path}")
    print(f"WandB run: {project_name}/{run_name}")
    print(f"\nTo use the model:")
    print(f"  from peft import PeftModel")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_path}')")
    
    # Finish WandB run
    wandb.finish()
    
    return trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train judge model on MPS (Apple Silicon)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-1.5B-Instruct",
        choices=[
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen1.5-4B-Chat",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        help="Model to fine-tune (Qwen2 1.5B Instruct default)"
    )
    parser.add_argument("--output_dir", default="./adapters_eval", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length (512 for 16GB MPS)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples (for testing)")
    parser.add_argument("--project_name", default="EN_PT_TRANSLATION_LORA", help="WandB project name")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("JUDGE MODEL TRAINING - MPS OPTIMIZED")
    print("="*80)
    print(f"System: M1 Pro (16GB RAM)")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation}")
    print(f"Effective batch: {args.batch_size * args.gradient_accumulation}")
    print("="*80 + "\n")
    
    trainer = train_judge_mps(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        project_name=args.project_name,
    )
