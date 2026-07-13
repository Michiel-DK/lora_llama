#!/usr/bin/env python3
"""Run evaluation only on trained adapter"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_train_cuda import evaluate_judge_model, load_judge_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import argparse
import wandb
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained judge model")
    parser.add_argument("--adapter_path", required=True, help="Path to trained adapter (e.g., adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20251209_150013_final)")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--test_samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--project_name", default="EN_PT_TRANSLATION_LORA", help="WandB project name")
    
    args = parser.parse_args()
    
    print("="*80)
    print("RUNNING EVALUATION ONLY")
    print("="*80)
    print(f"Adapter: {args.adapter_path}")
    print(f"Base model: {args.base_model}")
    
    # Initialize WandB
    adapter_name = os.path.basename(args.adapter_path)
    run_name = f"eval_{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nInitializing WandB...")
    wandb.init(
        project=args.project_name,
        name=run_name,
        config={
            "adapter_path": args.adapter_path,
            "base_model": args.base_model,
            "test_samples": args.test_samples,
            "max_seq_length": args.max_seq_length,
            "eval_only": True,
        }
    )
    print(f"✅ Logging to WandB: {args.project_name}/{run_name}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    _, _, test_dataset = load_judge_datasets()
    
    from judge_train_cuda import formatting_func
    test_dataset = test_dataset.map(formatting_func)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load base model - FORCE GPU with device_map
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0},  # Force GPU 0 instead of "auto"
    )
    print(f"✅ Base model on device: {base_model.device}")
    
    # Load adapter
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print(f"✅ Model with adapter on device: {next(model.parameters()).device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Verify model is actually on GPU before eval
    print(f"\n🔍 PRE-EVAL CHECK:")
    print(f"   Model device: {next(model.parameters()).device}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_judge_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        max_seq_length=args.max_seq_length,
        test_samples=args.test_samples,
    )
    
    # Log results to WandB
    wandb.log({
        "test/mae": results['mae'],
        "test/rmse": results['rmse'],
        "test/cohen_kappa": results['cohen_kappa'],
        "test/pearson_r": results['pearson_r'],
        "test/rouge_l_f1": results['rouge_l_f1'],
        "test/n_samples": results['n_samples'],
        "test/n_failed": results['n_failed'],
    })
    
    # Save results locally
    output_file = f"{args.adapter_path}/eval_results.json"
    with open(output_file, 'w') as f:
        # Remove predictions list for cleaner summary
        summary = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save predictions locally
    pred_file = f"{args.adapter_path}/test_predictions.json"
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(results['predictions'], f, ensure_ascii=False, indent=2)
    print(f"✅ Predictions saved to: {pred_file}")
    
    # Save predictions as WandB artifact
    artifact = wandb.Artifact(
        name=f"test_predictions_{adapter_name}",
        type="predictions",
        description=f"Test set predictions for {adapter_name}"
    )
    artifact.add_file(pred_file)
    wandb.log_artifact(artifact)
    print(f"✅ Saved predictions to WandB artifacts")
    
    wandb.finish()
    print(f"\n✅ WandB logging complete: {args.project_name}/{run_name}")

if __name__ == "__main__":
    main()
