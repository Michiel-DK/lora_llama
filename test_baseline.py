"""
Quick baseline model test - evaluate base model without any LoRA adapters.

This is a lightweight script to quickly test the base Llama model's
translation performance without training or fine-tuning.

Logs to WandB under the same project/run as strategy comparison.
"""

import params
import wandb
import weave
from datetime import datetime
from pt_app.trainer.trainer_pt import UniversalTrainer
from pt_app.data.dataset import LanguageDS


def test_baseline(
    max_samples: int = 50,
    generation_strategy: str = "beam_search",
    run_name: str = None
):
    """
    Test base model without any LoRA adapters.
    
    Args:
        max_samples: Number of samples to test (default: 50 for speed)
        generation_strategy: Strategy to use (default: beam_search)
        run_name: Optional custom run name for WandB
    """
    
    # Initialize WandB
    if run_name is None:
        run_name = f"baseline_model_{generation_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=params.PROJECT_NAME,
        name=run_name,
        tags=["baseline", "strategy_comparison", generation_strategy],
        config={
            "model": "baseline",
            "adapter_path": None,
            "max_samples": max_samples,
            "generation_strategy": generation_strategy,
            "is_baseline": True,
        }
    )
    
    weave.init(params.PROJECT_NAME)
    
    print("="*80)
    print("TESTING BASELINE MODEL (No LoRA adapters)")
    print("="*80)
    print(f"Strategy: {generation_strategy}")
    print(f"Max samples: {max_samples if max_samples else 'all'}")
    print("="*80 + "\n")
    
    # Initialize trainer WITHOUT LoRA
    trainer = UniversalTrainer()
    model, tokenizer = trainer.get_model(apply_lora=False)  # ← No LoRA!
    
    # Load test data
    print(f"[INFO] Loading test dataset: {params.DATASET}")
    _, _, test = LanguageDS(
        tokenizer=tokenizer,
        dataset=params.DATASET
    ).create_datasets(save=False)
    
    print(f"[INFO] Test dataset size: {len(test)}")
    if max_samples:
        print(f"[INFO] Limiting to {max_samples} samples for speed")
    
    # Test base model
    baseline_results = trainer.test_generation(
        adapter_path=None,  # ← No adapter = pure base model
        test_dataset=test,
        max_samples=max_samples,
        generation_strategy=generation_strategy,
        use_quality_filter=True,
        verbose_filter=False
    )
    
    # Log to WandB (matching strategy comparison format)
    wandb.log({
        f"baseline_{generation_strategy}/bleu": baseline_results['metrics'].get('bleu', 0),
        f"baseline_{generation_strategy}/rouge1": baseline_results['metrics'].get('rouge1_f1', 0),
        f"baseline_{generation_strategy}/rouge2": baseline_results['metrics'].get('rouge2_f1', 0),
        f"baseline_{generation_strategy}/rougeL": baseline_results['metrics'].get('rougeL_f1', 0),
        f"baseline_{generation_strategy}/perplexity": baseline_results['avg_perplexity'],
        f"baseline_{generation_strategy}/filter_pass_rate": baseline_results['filter_stats']['pass_rate'] if baseline_results['filter_stats'] else 1.0,
    })
    
    # Also log with generic "baseline" prefix for easy filtering
    wandb.log({
        "baseline/bleu": baseline_results['metrics'].get('bleu', 0),
        "baseline/rouge1": baseline_results['metrics'].get('rouge1_f1', 0),
        "baseline/rouge2": baseline_results['metrics'].get('rouge2_f1', 0),
        "baseline/rougeL": baseline_results['metrics'].get('rougeL_f1', 0),
        "baseline/perplexity": baseline_results['avg_perplexity'],
        "baseline/filter_pass_rate": baseline_results['filter_stats']['pass_rate'] if baseline_results['filter_stats'] else 1.0,
    })
    
    # Print results
    print("\n" + "="*80)
    print("BASELINE MODEL RESULTS")
    print("="*80)
    print(f"BLEU Score:        {baseline_results['metrics'].get('bleu', 0):.2f}")
    print(f"ROUGE-1 F1:        {baseline_results['metrics'].get('rouge1_f1', 0):.4f}")
    print(f"ROUGE-2 F1:        {baseline_results['metrics'].get('rouge2_f1', 0):.4f}")
    print(f"ROUGE-L F1:        {baseline_results['metrics'].get('rougeL_f1', 0):.4f}")
    print(f"Perplexity:        {baseline_results['avg_perplexity']:.2f}")
    
    if baseline_results['filter_stats']:
        print(f"Filter Pass Rate:  {baseline_results['filter_stats']['pass_rate']*100:.1f}%")
    
    print("="*80)
    
    # Show a few examples
    print("\nFirst 3 Translation Examples:")
    print("-"*80)
    for i, example in enumerate(baseline_results['examples'][:3]):
        print(f"\nExample {i+1}:")
        print(f"  Input:  {example['input'][:100]}...")
        print(f"  Output: {example['raw_output'][:100]}...")
        print(f"  Reference: {example['reference'][:100]}...")
    print("-"*80)
    
    wandb.finish()
    
    return baseline_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test baseline model")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum samples to test (default: 50, use 0 for all)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["greedy", "beam_search", "sampling"],
        default="beam_search",
        help="Generation strategy (default: beam_search)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom WandB run name (optional)"
    )
    
    args = parser.parse_args()
    
    # Run baseline test
    results = test_baseline(
        max_samples=args.max_samples if args.max_samples > 0 else None,
        generation_strategy=args.strategy,
        run_name=args.run_name
    )
    
    print("\n✅ Baseline test complete! Check WandB for results.")
    print(f"💡 Logged to WandB with tags: ['baseline', 'strategy_comparison', '{args.strategy}']")
    print(f"💡 To compare with fine-tuned model, run:")
    print(f"   python compare_baseline_vs_finetuned.py --adapter ./adapters/YOUR_ADAPTER --max-samples {args.max_samples}")
