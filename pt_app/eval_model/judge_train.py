# train_judge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import json

def load_judge_datasets():
    """Load formatted train/val datasets"""
    
    # Load from JSON
    with open("judge_train.json", "r") as f:
        train_data = json.load(f)
    
    with open("judge_val.json", "r") as f:
        val_data = json.load(f)
    
    # Convert to HuggingFace Dataset
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def train_judge(
    output_dir="./qwen-3b-judge",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
):
    """
    Fine-tune Qwen 3B as translation judge
    """
    
    print("="*60)
    print("TRAINING QWEN 3B JUDGE")
    print("="*60)
    
    # Load model
    print("\n1. Loading Qwen 3B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for QLoRA
    print("\n2. Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # Load datasets
    print("\n3. Loading training data...")
    train_dataset, val_dataset = load_judge_datasets()
    
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        load_best_model_at_end=True,
    )
    
    # Trainer
    print("\n4. Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=768,
        dataset_text_field="messages",  # Will use chat template
    )
    
    # Train
    print("\n5. Starting training...")
    print("="*60)
    trainer.train()
    
    # Save final model
    print("\n6. Saving model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {output_dir}/final")
    
    return trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./qwen-3b-judge", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    trainer = train_judge(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )