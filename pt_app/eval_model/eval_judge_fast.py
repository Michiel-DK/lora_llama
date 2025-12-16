#!/usr/bin/env python3
"""Fast evaluation with detailed progress tracking"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_train_cuda import load_judge_datasets, formatting_func, extract_score, extract_feedback
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import argparse
import wandb
from datetime import datetime
import time
from rouge_score import rouge_scorer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr

def evaluate_fast(model, tokenizer, test_dataset, device, max_seq_length=512, test_samples=None):
    """Fast evaluation with detailed progress"""
    print("\n" + "="*80)
    print("FAST EVALUATION WITH PROGRESS TRACKING")
    print("="*80)
    
    model.eval()
    
    if test_samples:
        test_dataset = test_dataset.select(range(min(test_samples, len(test_dataset))))
    
    print(f"Testing on {len(test_dataset)} samples")
    print(f"Model device: {next(model.parameters()).device}")
    
    predicted_scores = []
    reference_scores = []
    predicted_feedbacks = []
    reference_feedbacks = []
    examples = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print("\nGenerating predictions...")
    start_time = time.time()
    
    for i, example in enumerate(test_dataset):
        iter_start = time.time()
        
        print(f"\n🔍 Processing example {i}/{len(test_dataset)}...")
        
        messages = example["messages"]
        user_prompt = messages[0]["content"]
        reference_response = messages[1]["content"]
        
        print(f"   Prompt length: {len(user_prompt)} chars")
        
        input_text = f"User: {user_prompt}\n\nAssistant:"
        
        # Tokenize and move to device
        print(f"   Tokenizing...")
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        print(f"   Input tokens: {inputs['input_ids'].shape[1]}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"   Inputs moved to: {inputs['input_ids'].device}")
        
        # Generate with minimal settings
        print(f"   🚀 Starting generation (max 128 tokens)...")
        gen_start = time.time()
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=512,  # Increased to allow full responses with score
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,  # Explicit greedy
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            gen_time = time.time() - gen_start
            print(f"   ✓ Generation done in {gen_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            continue
        
        print(f"   Decoding output...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_response = generated_text.split("Assistant:")[-1].strip()
        
        print(f"   Generated: {predicted_response[:100]}...")
        
        pred_score = extract_score(predicted_response)
        ref_score = extract_score(reference_response)
        
        pred_feedback = extract_feedback(predicted_response)
        ref_feedback = extract_feedback(reference_response)
        
        iter_time = time.time() - iter_start
        print(f"   ✅ Example {i} complete in {iter_time:.2f}s (pred={pred_score}, ref={ref_score})")
        
        example_dict = {
            'user_prompt': user_prompt,
            'predicted_response': predicted_response,
            'reference_response': reference_response,
            'predicted_score': pred_score,
            'reference_score': ref_score,
            'score_extracted': pred_score is not None and ref_score is not None,
            'time_taken': iter_time,
        }
        examples.append(example_dict)
        
        if pred_score is not None and ref_score is not None:
            predicted_scores.append(pred_score)
            reference_scores.append(ref_score)
            predicted_feedbacks.append(pred_feedback)
            reference_feedbacks.append(ref_feedback)
    
    total_time = time.time() - start_time
    print(f"\n✓ Generated {len(predicted_scores)}/{len(test_dataset)} valid predictions in {total_time:.1f}s")
    print(f"  Average: {total_time/len(test_dataset):.2f}s per sample")
    
    # Calculate metrics
    predicted_scores = np.array(predicted_scores)
    reference_scores = np.array(reference_scores)
    
    mae = mean_absolute_error(reference_scores, predicted_scores)
    rmse = np.sqrt(mean_squared_error(reference_scores, predicted_scores))
    
    pred_rounded = np.round(predicted_scores).astype(int)
    ref_rounded = np.round(reference_scores).astype(int)
    kappa = cohen_kappa_score(ref_rounded, pred_rounded)
    
    # Pearson correlation (handle constant/insufficient data)
    if len(predicted_scores) < 2:
        pearson_r = np.nan
        print("⚠️  Not enough samples for correlation")
    elif np.std(predicted_scores) == 0 or np.std(reference_scores) == 0:
        pearson_r = np.nan
        print("⚠️  Constant values - correlation undefined")
    else:
        pearson_r, _ = pearsonr(predicted_scores, reference_scores)
    
    rouge_scores = []
    for pred_fb, ref_fb in zip(predicted_feedbacks, reference_feedbacks):
        score = scorer.score(ref_fb, pred_fb)
        rouge_scores.append(score['rougeL'].fmeasure)
    rouge_l_f1 = np.mean(rouge_scores) if rouge_scores else 0.0
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'cohen_kappa': kappa,
        'pearson_r': pearson_r,
        'rouge_l_f1': rouge_l_f1,
        'n_samples': len(predicted_scores),
        'n_failed': len(test_dataset) - len(predicted_scores),
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(test_dataset),
        'predictions': examples,
    }
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | Kappa: {kappa:.3f} | Pearson: {pearson_r:.3f} | ROUGE-L: {rouge_l_f1:.3f}")
    print("="*80)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--project_name", default="EN_PT_TRANSLATION_LORA")
    
    args = parser.parse_args()
    
    print(f"Adapter: {args.adapter_path}")
    print(f"Base model: {args.base_model}")
    
    # WandB
    adapter_name = os.path.basename(args.adapter_path)
    run_name = f"eval_{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))
    
    # Load data
    _, _, test_dataset = load_judge_datasets()
    test_dataset = test_dataset.map(formatting_func)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model - FORCE GPU
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map={"": 0},  # Force GPU 0
    )
    
    # Clear problematic generation config
    if hasattr(base_model, 'generation_config'):
        base_model.generation_config.temperature = None
        base_model.generation_config.top_p = None
        base_model.generation_config.top_k = None
        base_model.generation_config.do_sample = False
        print("✓ Cleared generation config")
    
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate
    results = evaluate_fast(model, tokenizer, test_dataset, device, args.max_seq_length, args.test_samples)
    
    # Save and log
    pred_file = f"{args.adapter_path}/test_predictions_fast.json"
    with open(pred_file, 'w') as f:
        json.dump(results['predictions'], f, indent=2)
    
    wandb.log({
        "test/mae": results['mae'],
        "test/rmse": results['rmse'],
        "test/cohen_kappa": results['cohen_kappa'],
        "test/pearson_r": results['pearson_r'],
        "test/rouge_l_f1": results['rouge_l_f1'],
        "test/total_time": results['total_time'],
        "test/avg_time_per_sample": results['avg_time_per_sample'],
    })
    
    artifact = wandb.Artifact(f"predictions_{adapter_name}", type="predictions")
    artifact.add_file(pred_file)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print(f"\n✅ Done! Results in {pred_file}")

if __name__ == "__main__":
    main()
