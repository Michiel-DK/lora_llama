"""
Compare base model performance vs fine-tuned model performance.

This script evaluates:
1. Base Llama model (no adapters)
2. Fine-tuned model (with LoRA adapters)

Shows the improvement from fine-tuning.
"""

import params
import wandb
import weave
import torch
from datetime import datetime

from pt_app.trainer.trainer_pt import UniversalTrainer
from pt_app.data.dataset import LanguageDS


def compare_baseline_vs_finetuned(
    adapter_path: str,
    max_samples: int = None,
    generation_strategy: str = "beam_search"
):
    """
    Compare base model vs fine-tuned model performance.
    
    Args:
        adapter_path: Path to fine-tuned adapter
        max_samples: Maximum samples to evaluate (None = all)
        generation_strategy: Strategy to use (greedy, beam_search, sampling)
    """
    
    run_name = f"baseline_vs_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=params.PROJECT_NAME,
        name=run_name,
        tags=["baseline_comparison", "evaluation"],
        config={
            "adapter_path": adapter_path,
            "max_samples": max_samples,
            "generation_strategy": generation_strategy,
            "dataset": params.DATASET,
            "model": params.MODEL_NAME,
        }
    )
    
    weave.init(params.PROJECT_NAME)
    
    print("="*80)
    print("BASELINE vs FINE-TUNED MODEL COMPARISON")
    print("="*80)
    
    # ============================================================================
    # STEP 1: Test BASE MODEL (no adapters)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: Testing BASE MODEL (no LoRA adapters)")
    print("="*80)
    
    trainer_baseline = UniversalTrainer()
    # Force CPU to avoid MPS issues with baseline model
    trainer_baseline.device = torch.device("cpu")
    trainer_baseline.device_type = "cpu"
    model, tokenizer = trainer_baseline.get_model(apply_lora=False)  # ← Load WITHOUT LoRA
    
    # Load test dataset
    print(f"[INFO] Loading test dataset: {params.DATASET}")
    _, _, test_dataset = LanguageDS(
        tokenizer=tokenizer,
        dataset=params.DATASET,
    ).create_datasets(save=False)
    
    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    if max_samples:
        print(f"[INFO] Limiting to {max_samples} samples")
    
    baseline_results = trainer_baseline.test_generation(
        adapter_path=None,  # ← No adapter = base model
        test_dataset=test_dataset,
        max_samples=max_samples,
        use_quality_filter=True,
        verbose_filter=False,
        generation_strategy=generation_strategy
    )
    
    # ============================================================================
    # STEP 2: Test FINE-TUNED MODEL (with adapters)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: Testing FINE-TUNED MODEL (with LoRA adapters)")
    print("="*80)
    
    trainer_finetuned = UniversalTrainer()
    model, tokenizer = trainer_finetuned.get_model(apply_lora=True)  # ← Load WITH LoRA (for initialization)
    
    finetuned_results = trainer_finetuned.test_generation(
        adapter_path=adapter_path,  # ← Load trained adapter
        test_dataset=test_dataset,
        max_samples=max_samples,
        use_quality_filter=True,
        verbose_filter=False,
        generation_strategy=generation_strategy
    )
    
    # ============================================================================
    # STEP 3: Calculate Improvements
    # ============================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    baseline_bleu = baseline_results['metrics'].get('bleu', 0)
    finetuned_bleu = finetuned_results['metrics'].get('bleu', 0)
    bleu_improvement = finetuned_bleu - baseline_bleu
    bleu_improvement_pct = (bleu_improvement / baseline_bleu * 100) if baseline_bleu > 0 else 0
    
    baseline_rouge = baseline_results['metrics'].get('rougeL_f1', 0)
    finetuned_rouge = finetuned_results['metrics'].get('rougeL_f1', 0)
    rouge_improvement = finetuned_rouge - baseline_rouge
    rouge_improvement_pct = (rouge_improvement / baseline_rouge * 100) if baseline_rouge > 0 else 0
    
    baseline_perplexity = baseline_results['avg_perplexity']
    finetuned_perplexity = finetuned_results['avg_perplexity']
    perplexity_improvement = baseline_perplexity - finetuned_perplexity  # Lower is better
    perplexity_improvement_pct = (perplexity_improvement / baseline_perplexity * 100) if baseline_perplexity > 0 else 0
    
    # Create comparison table
    comparison_table = wandb.Table(
        columns=["Model", "BLEU", "ROUGE-L", "Perplexity", "Pass Rate"],
        data=[
            [
                "Baseline",
                f"{baseline_bleu:.2f}",
                f"{baseline_rouge:.4f}",
                f"{baseline_perplexity:.2f}",
                f"{baseline_results['filter_stats']['pass_rate']*100:.1f}%" if baseline_results['filter_stats'] else "100.0%"
            ],
            [
                "Fine-tuned",
                f"{finetuned_bleu:.2f}",
                f"{finetuned_rouge:.4f}",
                f"{finetuned_perplexity:.2f}",
                f"{finetuned_results['filter_stats']['pass_rate']*100:.1f}%" if finetuned_results['filter_stats'] else "100.0%"
            ],
            [
                "Improvement",
                f"{bleu_improvement:+.2f} ({bleu_improvement_pct:+.1f}%)",
                f"{rouge_improvement:+.4f} ({rouge_improvement_pct:+.1f}%)",
                f"{perplexity_improvement:+.2f} ({perplexity_improvement_pct:+.1f}%)",
                "-"
            ]
        ]
    )
    
    # Log only the comparison table (no individual metrics)
    wandb.log({
        "comparison_summary": comparison_table,
        "samples_tested": len(test_dataset) if max_samples is None else min(max_samples, len(test_dataset))
    })
    
    # Print comparison
    print("\nMetric Comparison:")
    print("-" * 80)
    print(f"{'Model':<15} {'BLEU':>8} {'ROUGE-L':>10} {'Perplexity':>12} {'Pass Rate':>10}")
    print("-" * 80)
    print(f"{'Baseline':<15} {baseline_bleu:>8.2f} {baseline_rouge:>10.4f} {baseline_perplexity:>12.2f} {baseline_results['filter_stats']['pass_rate']*100 if baseline_results['filter_stats'] else 100.0:>9.1f}%")
    print(f"{'Fine-tuned':<15} {finetuned_bleu:>8.2f} {finetuned_rouge:>10.4f} {finetuned_perplexity:>12.2f} {finetuned_results['filter_stats']['pass_rate']*100 if finetuned_results['filter_stats'] else 100.0:>9.1f}%")
    print("-" * 80)
    print(f"{'Improvement':<15} {bleu_improvement:>+8.2f} {rouge_improvement:>+10.4f} {perplexity_improvement:>+12.2f}")
    print(f"{'(Percentage)':<15} {bleu_improvement_pct:>+7.1f}% {rouge_improvement_pct:>+9.1f}% {perplexity_improvement_pct:>+11.1f}%")
    print("-" * 80)
    
    print("\n" + "="*80)
    if bleu_improvement > 0:
        print(f"✅ Fine-tuning improved BLEU by {bleu_improvement:.2f} points ({bleu_improvement_pct:.1f}%)")
    else:
        print(f"⚠️  Fine-tuning decreased BLEU by {abs(bleu_improvement):.2f} points ({abs(bleu_improvement_pct):.1f}%)")
    
    if rouge_improvement > 0:
        print(f"✅ Fine-tuning improved ROUGE-L by {rouge_improvement:.4f} points ({rouge_improvement_pct:.1f}%)")
    else:
        print(f"⚠️  Fine-tuning decreased ROUGE-L by {abs(rouge_improvement):.4f} points ({abs(rouge_improvement_pct):.1f}%)")
    
    if perplexity_improvement > 0:
        print(f"✅ Fine-tuning improved perplexity by {perplexity_improvement:.2f} points ({perplexity_improvement_pct:.1f}%)")
    else:
        print(f"⚠️  Fine-tuning increased perplexity by {abs(perplexity_improvement):.2f} points ({abs(perplexity_improvement_pct):.1f}%)")
    print("="*80)
    
    wandb.finish()
    
    return {
        'baseline': baseline_results,
        'finetuned': finetuned_results,
        'improvements': {
            'bleu': bleu_improvement,
            'bleu_pct': bleu_improvement_pct,
            'rouge_l': rouge_improvement,
            'rouge_l_pct': rouge_improvement_pct,
            'perplexity': perplexity_improvement,
            'perplexity_pct': perplexity_improvement_pct,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to fine-tuned adapter"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "beam_search", "sampling"],
        default="beam_search",
        help="Generation strategy (default: beam_search)"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_baseline_vs_finetuned(
        adapter_path=args.adapter,
        max_samples=args.max_samples,
        generation_strategy=args.strategy
    )
    
    print("\n✅ Comparison complete! Check WandB for detailed results.")
