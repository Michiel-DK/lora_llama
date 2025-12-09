# train_judge_cuda.py - Optimized for CUDA GPUs (Vast.ai)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
from datetime import datetime
import wandb
import weave
import re
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from rouge_score import rouge_scorer

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

def extract_score(text):
    """Extract numeric score from judge output (e.g., 'Score: 8/10' -> 8.0)"""
    match = re.search(r'Score:\s*(\d+(?:\.\d+)?)\s*/\s*10', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def extract_feedback(text):
    """Extract feedback text after 'Feedback:' """
    match = re.search(r'Feedback:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return full text if no feedback marker found

def evaluate_judge_model(model, tokenizer, test_dataset, device, max_seq_length=512, test_samples=None):
    """
    Evaluate judge model on test set with comprehensive metrics
    
    Returns dict with:
    - mae: Mean Absolute Error of scores
    - rmse: Root Mean Squared Error
    - cohen_kappa: Agreement score (0-1)
    - pearson_r: Correlation coefficient
    - rouge_l_f1: ROUGE-L F1 for feedback text
    - predictions: List of (predicted, reference, example) tuples
    """
    print("\n" + "="*80)
    print("EVALUATING JUDGE MODEL ON TEST SET")
    print("="*80)
    
    model.eval()
    
    # Limit test samples if specified
    if test_samples:
        test_dataset = test_dataset.select(range(min(test_samples, len(test_dataset))))
        print(f"Testing on {len(test_dataset)} samples")
    else:
        print(f"Testing on all {len(test_dataset)} samples")
    
    predicted_scores = []
    reference_scores = []
    predicted_feedbacks = []
    reference_feedbacks = []
    examples = []
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print("\nGenerating predictions...")
    for i, example in enumerate(test_dataset):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_dataset)}")
        
        # Get user prompt and reference response
        messages = example["messages"]
        user_prompt = messages[0]["content"]
        reference_response = messages[1]["content"]
        
        # Format input
        input_text = f"User: {user_prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate (greedy decoding for consistency)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding - removes temp/top_p warnings
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant response
        predicted_response = generated_text.split("Assistant:")[-1].strip()
        
        # Extract scores
        pred_score = extract_score(predicted_response)
        ref_score = extract_score(reference_response)
        
        # Extract feedbacks
        pred_feedback = extract_feedback(predicted_response)
        ref_feedback = extract_feedback(reference_response)
        
        # Store ALL examples (even if score extraction fails)
        example_dict = {
            'user_prompt': user_prompt,
            'predicted_response': predicted_response,
            'reference_response': reference_response,
            'predicted_score': pred_score,
            'reference_score': ref_score,
            'score_extracted': pred_score is not None and ref_score is not None,
        }
        examples.append(example_dict)
        
        # Only add to metrics if both scores extracted successfully
        if pred_score is not None and ref_score is not None:
            predicted_scores.append(pred_score)
            reference_scores.append(ref_score)
            predicted_feedbacks.append(pred_feedback)
            reference_feedbacks.append(ref_feedback)
    
    print(f"✓ Generated {len(predicted_scores)}/{len(test_dataset)} valid predictions\n")
    
    failed_extractions = len(test_dataset) - len(predicted_scores)
    if failed_extractions > 0:
        print(f"⚠️  Warning: {failed_extractions} examples failed score extraction")
        print(f"   Check that model outputs match 'Score: X/10' format")
    
    # Calculate metrics
    predicted_scores = np.array(predicted_scores)
    reference_scores = np.array(reference_scores)
    
    # Score metrics
    mae = mean_absolute_error(reference_scores, predicted_scores)
    rmse = np.sqrt(mean_squared_error(reference_scores, predicted_scores))
    
    # Cohen's Kappa (need to round to integers for kappa)
    pred_rounded = np.round(predicted_scores).astype(int)
    ref_rounded = np.round(reference_scores).astype(int)
    kappa = cohen_kappa_score(ref_rounded, pred_rounded)
    
    # Pearson correlation
    pearson_r, _ = pearsonr(predicted_scores, reference_scores)
    
    # ROUGE-L for feedback text
    rouge_scores = []
    for pred_fb, ref_fb in zip(predicted_feedbacks, reference_feedbacks):
        score = scorer.score(ref_fb, pred_fb)
        rouge_scores.append(score['rougeL'].fmeasure)
    rouge_l_f1 = np.mean(rouge_scores)
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'cohen_kappa': kappa,
        'pearson_r': pearson_r,
        'rouge_l_f1': rouge_l_f1,
        'n_samples': len(predicted_scores),
        'n_failed': failed_extractions,
        'predictions': examples,  # Store ALL examples (not just first 10)
    }
    
    # Print results
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Samples evaluated: {results['n_samples']}")
    print(f"\nScore Metrics:")
    print(f"  MAE:             {mae:.3f} points")
    print(f"  RMSE:            {rmse:.3f} points")
    print(f"  Cohen's Kappa:   {kappa:.3f} (agreement)")
    print(f"  Pearson r:       {pearson_r:.3f} (correlation)")
    print(f"\nFeedback Quality:")
    print(f"  ROUGE-L F1:      {rouge_l_f1:.3f}")
    print("="*80)
    
    # Show a few examples
    print("\nExample Predictions (first 3):")
    for i, ex in enumerate(examples[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Reference Score: {ex['reference_score']:.1f}/10")
        print(f"Predicted Score: {ex['predicted_score']:.1f}/10")
        print(f"Error: {abs(ex['predicted_score'] - ex['reference_score']):.1f}")
    
    return results

def train_judge_cuda(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # Qwen2.5 3B Instruct for judge training
    output_dir="./adapters_eval",
    num_epochs=3,
    batch_size=2,
    gradient_accumulation_steps=2,  # Effective batch size = 4
    learning_rate=2e-4,
    max_seq_length=512,
    max_samples=None,  # Limit dataset size for testing
    test_samples=None,  # Limit test dataset size
    project_name="EN_PT_TRANSLATION_LORA",
    use_4bit=True,  # Use QLoRA 4-bit quantization
    skip_eval=False,  # Skip test evaluation
):
    """
    Fine-tune model as translation judge - OPTIMIZED FOR CUDA (Vast.ai)
    
    Default config for Qwen2.5 3B Instruct on 8-12GB GPU:
    - batch_size=2, gradient_accumulation_steps=2 (effective batch=4)
    - max_seq_length=512
    - 4-bit QLoRA quantization (fits in 8GB VRAM)
    - fp16 precision, paged_adamw_8bit optimizer
    - WandB + Weave logging enabled
    """
    
    # Initialize WandB and Weave
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    run_name = f"{model_short}-judge-{num_epochs}ep-{timestamp}"
    
    print("="*80)
    print(f"TRAINING JUDGE MODEL ON CUDA (Vast.ai GPU)")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Run name: {run_name}")
    print(f"Quantization: {'4-bit QLoRA' if use_4bit else 'fp16'}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*80)
    
    # Initialize logging
    print("\nInitializing WandB and Weave...")
    wandb.init(
        project=project_name,
        name=run_name,
        tags=["judge_training", "cuda", "lora", "vast.ai"],
        config={
            "model": model_name,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "device": "cuda",
            "use_4bit": use_4bit,
        }
    )
    
    weave.init(project_name)
    print(f"✅ Logging to WandB: {project_name}/{run_name}")
    print(f"✅ Weave initialized: {project_name}")
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print("⚠️  CUDA not available, using CPU (will be VERY slow)")
    
    # Load model with 4-bit quantization for memory efficiency
    print("\n1. Loading model...")
    
    if use_4bit:
        # QLoRA 4-bit config for 8GB GPUs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir="./cache/",
            device_map="auto",
            use_cache=False,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        print("✅ Model loaded with 4-bit quantization")
    else:
        # Standard fp16 loading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir="./cache/",
            device_map="auto",
            use_cache=False,
        )
        print("✅ Model loaded with fp16")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./cache/",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA configuration
    print("\n2. Applying LoRA adapters...")
    
    # Adjust LoRA rank based on model size
    if "1B" in model_name or "1.5B" in model_name or "1.8B" in model_name:
        lora_r = 16  # Higher rank for smaller models (1B-2B)
        lora_alpha = 32
    elif "3B" in model_name or "4B" in model_name:
        lora_r = 16  # Medium rank for 3B models
        lora_alpha = 32
    elif "7B" in model_name:
        lora_r = 8  # Smaller for 7B
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
    # No need to move to device - device_map="auto" handles it
    
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
    
    print(f"\n4. Training configuration:")
    print(f"   Per-device batch size: {per_device_batch}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Gradient checkpointing: Enabled")
    
    # Training arguments - CUDA OPTIMIZED with WandB integration
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        report_to=["wandb"],
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
        
        # CUDA-specific settings
        fp16=True,  # Use mixed precision for speed
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        # 8-bit optimizer for memory efficiency
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        
        # Memory management
        gradient_checkpointing=True,  # Enable for memory efficiency
        max_grad_norm=0.3,  # Lower for stability with 4-bit
    )
    
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\n7. Starting training...")
    print("="*80)
    print("💡 TIP: Monitor VRAM usage with: watch -n 1 nvidia-smi")
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
    
    # Finish WandB run if skipping eval
    if args.skip_eval:
        print("\n9. Skipping test evaluation (--skip_eval flag set)")
        wandb.finish()
        return trainer, None
    
    # Evaluate on test set
    print("\n9. Running evaluation on test set...")
    
    # Clear CUDA cache before eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Keep model on GPU if available (faster), otherwise CPU
    if torch.cuda.is_available():
        eval_device = torch.device("cuda")
        print("[INFO] Running eval on GPU (faster)")
    else:
        model = model.to("cpu")
        eval_device = torch.device("cpu")
        print("[INFO] Running eval on CPU")
    
    # Run evaluation
    eval_results = evaluate_judge_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=eval_device,
        max_seq_length=max_seq_length,
        test_samples=test_samples,
    )
    
    # Save predictions to JSON file
    predictions_file = f"{final_path}/test_predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results['predictions'], f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(eval_results['predictions'])} predictions to: {predictions_file}")
    
    # Log results to WandB
    wandb.log({
        "test/mae": eval_results['mae'],
        "test/rmse": eval_results['rmse'],
        "test/cohen_kappa": eval_results['cohen_kappa'],
        "test/pearson_r": eval_results['pearson_r'],
        "test/rouge_l_f1": eval_results['rouge_l_f1'],
        "test/n_samples": eval_results['n_samples'],
        "test/n_failed": eval_results['n_failed'],
    })
    
    # Save predictions as WandB artifact
    artifact = wandb.Artifact(
        name=f"test_predictions_{run_name}",
        type="predictions",
        description=f"Test set predictions for {model_name}"
    )
    artifact.add_file(predictions_file)
    wandb.log_artifact(artifact)
    print(f"✅ Saved predictions to WandB artifacts")
    
    print(f"\nTo use the model:")
    print(f"  from peft import PeftModel")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_path}')")
    
    # Finish WandB run
    wandb.finish()
    
    return trainer, eval_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train judge model on CUDA (Vast.ai)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        choices=[
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen1.5-4B-Chat",
            "Qwen/Qwen2.5-3B-Instruct",
        ],
        help="Model to fine-tune (Qwen2.5 3B Instruct default)"
    )
    parser.add_argument("--output_dir", default="./adapters_eval", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples (for testing)")
    parser.add_argument("--test_samples", type=int, default=None, help="Limit number of test samples (for faster eval)")
    parser.add_argument("--project_name", default="EN_PT_TRANSLATION_LORA", help="WandB project name")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (use fp16)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip test evaluation after training")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("JUDGE MODEL TRAINING - CUDA OPTIMIZED")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.gradient_accumulation}")
    print(f"Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"Quantization: {'4-bit QLoRA' if not args.no_4bit else 'fp16'}")
    print("="*80 + "\n")
    
    trainer, eval_results = train_judge_cuda(
        model_name=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        test_samples=args.test_samples,
        project_name=args.project_name,
        use_4bit=not args.no_4bit,
        skip_eval=args.skip_eval,
    )
