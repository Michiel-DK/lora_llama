"""
Compare different generation strategies for translation quality.

This script runs evaluation with three different generation strategies:
1. Greedy decoding (deterministic, fastest)
2. Beam search (explores alternatives, slower but often better)
3. Sampling with low temperature (balanced approach)

Results are logged to WandB and Weave for comparison.
"""

import os
from datetime import datetime
import params
import wandb
import weave

from pt_app.trainer.trainer_pt import UniversalTrainer
from pt_app.data.dataset import LanguageDS


def compare_generation_strategies(
    adapter_path: str,
    test_dataset=None,
    max_samples: int = None,
    strategies: list = None
):
    """
    Compare multiple generation strategies on the same test set.
    
    Args:
        adapter_path: Path to trained adapter
        test_dataset: Test dataset (if None, will use params.DATASET)
        max_samples: Maximum samples to evaluate (None = all)
        strategies: List of strategy names to test (default: all three)
    """
    
    if strategies is None:
        strategies = ["greedy", "beam_search", "sampling"]
    
    # Initialize tracking
    run_name = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=params.PROJECT_NAME,
        name=run_name,
        tags=["strategy_comparison", "evaluation"],
        config={
            "adapter_path": adapter_path,
            "max_samples": max_samples,
            "strategies": strategies,
            "test_dataset": params.DATASET if test_dataset is None else "custom",
        }
    )
    
    weave.init(params.PROJECT_NAME)
    
    # Initialize trainer
    print("[INFO] Initializing trainer...")
    trainer = UniversalTrainer()
    model, tokenizer = trainer.get_model()
    
    # Load test dataset if not provided
    if test_dataset is None:
        print(f"[INFO] Loading test dataset: {params.DATASET}")
        _, _, test_dataset = LanguageDS(
            tokenizer=tokenizer,
            dataset=params.DATASET,
        ).create_datasets(save=False)
    
    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    if max_samples:
        print(f"[INFO] Limiting to {max_samples} samples")
    
    # Store all results for comparison
    all_results = {}
    
    # Test each strategy
    for strategy in strategies:
        print("\n" + "="*80)
        print(f"TESTING STRATEGY: {strategy.upper()}")
        print("="*80)
        
        results = trainer.test_generation(
            adapter_path=adapter_path,
            test_dataset=test_dataset,
            max_samples=max_samples,
            use_quality_filter=True,
            verbose_filter=False,
            generation_strategy=strategy
        )
        
        all_results[strategy] = results
        
        # Log strategy-specific metrics
        wandb.log({
            f"{strategy}/bleu": results['metrics'].get('bleu', 0),
            f"{strategy}/rouge1": results['metrics'].get('rouge1_f1', 0),
            f"{strategy}/rouge2": results['metrics'].get('rouge2_f1', 0),
            f"{strategy}/rougeL": results['metrics'].get('rougeL_f1', 0),
            f"{strategy}/perplexity": results['avg_perplexity'],
            f"{strategy}/filter_pass_rate": results['filter_stats']['pass_rate'] if results['filter_stats'] else 1.0,
        })
    
    # Create comparison summary
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    
    comparison_table = wandb.Table(
        columns=[
            "Strategy",
            "BLEU",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Perplexity",
            "Filter Pass Rate"
        ],
        data=[
            [
                strategy,
                f"{all_results[strategy]['metrics'].get('bleu', 0):.2f}",
                f"{all_results[strategy]['metrics'].get('rouge1_f1', 0):.4f}",
                f"{all_results[strategy]['metrics'].get('rouge2_f1', 0):.4f}",
                f"{all_results[strategy]['metrics'].get('rougeL_f1', 0):.4f}",
                f"{all_results[strategy]['avg_perplexity']:.2f}",
                f"{all_results[strategy]['filter_stats']['pass_rate']*100:.1f}%" if all_results[strategy]['filter_stats'] else "100.0%"
            ]
            for strategy in strategies
        ]
    )
    
    wandb.log({"strategy_comparison_table": comparison_table})
    
    # Print comparison
    print("\nMetric Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'BLEU':>8} {'ROUGE-L':>10} {'Perplexity':>12} {'Pass Rate':>10}")
    print("-" * 80)
    
    for strategy in strategies:
        bleu = all_results[strategy]['metrics'].get('bleu', 0)
        rouge_l = all_results[strategy]['metrics'].get('rougeL_f1', 0)
        perplexity = all_results[strategy]['avg_perplexity']
        pass_rate = all_results[strategy]['filter_stats']['pass_rate']*100 if all_results[strategy]['filter_stats'] else 100.0
        
        print(f"{strategy:<15} {bleu:>8.2f} {rouge_l:>10.4f} {perplexity:>12.2f} {pass_rate:>9.1f}%")
    
    print("-" * 80)
    
    # Find best strategy for each metric
    best_bleu = max(strategies, key=lambda s: all_results[s]['metrics'].get('bleu', 0))
    best_rouge = max(strategies, key=lambda s: all_results[s]['metrics'].get('rougeL_f1', 0))
    best_perplexity = min(strategies, key=lambda s: all_results[s]['avg_perplexity'])
    
    print("\nBest Strategies:")
    print(f"  BLEU: {best_bleu} ({all_results[best_bleu]['metrics'].get('bleu', 0):.2f})")
    print(f"  ROUGE-L: {best_rouge} ({all_results[best_rouge]['metrics'].get('rougeL_f1', 0):.4f})")
    print(f"  Perplexity: {best_perplexity} ({all_results[best_perplexity]['avg_perplexity']:.2f})")
    
    # Log best strategies
    wandb.log({
        "best_strategy_bleu": best_bleu,
        "best_strategy_rouge": best_rouge,
        "best_strategy_perplexity": best_perplexity,
    })
    
    print("="*80)
    
    wandb.finish()
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare generation strategies")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to trained adapter"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["greedy", "beam_search", "sampling"],
        default=None,
        help="Strategies to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_generation_strategies(
        adapter_path=args.adapter,
        max_samples=args.max_samples,
        strategies=args.strategies
    )
    
    print("\n✅ Comparison complete! Check WandB for detailed results.")
