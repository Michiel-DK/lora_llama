"""
Train the English->Portuguese LoRA translation adapter end to end.

Loads the base model, builds the datasets, runs a couple of sanity checks
(train/test leakage + chat-template formatting), trains the LoRA adapter, and
evaluates it on the held-out OPUS test set. All configuration lives in params.py.

Usage:
    python train.py
"""
import random

import weave

import params
from pt_app.trainer.trainer_pt import UniversalTrainer
from pt_app.data.dataset import LanguageDS


def get_source_text(item, tokenizer):
    """Return the English source text for a (possibly pre-tokenized) dataset item."""
    if 'source_text' in item:
        return item['source_text']
    input_ids = item['input_ids']
    labels = item['labels']
    label_start = next(i for i, l in enumerate(labels) if l != -100)
    source_ids = input_ids[:label_start]
    return tokenizer.decode(source_ids, skip_special_tokens=True)


def check_data_leakage(train, test, tokenizer):
    """Warn if any source sentence appears in both train and test."""
    print("\n" + "=" * 80)
    print("CHECKING FOR DATA LEAKAGE")
    print("=" * 80)

    train_sources = {get_source_text(train[i], tokenizer) for i in range(len(train))}
    test_sources = {get_source_text(test[i], tokenizer) for i in range(len(test))}
    overlap = train_sources & test_sources

    print(f"Train samples: {len(train_sources)}")
    print(f"Test samples: {len(test_sources)}")
    print(f"⚠️  OVERLAP: {len(overlap)} samples!")
    if overlap:
        print("\nFirst 5 overlapping samples:")
        for i, text in enumerate(list(overlap)[:5]):
            print(f"  {i + 1}. {text[:100]}...")
    print("=" * 80 + "\n")


def verify_data_quality(train, tokenizer, n_samples=10):
    """Spot-check chat-template formatting and that labels exclude the prompt."""
    print("\n" + "=" * 80)
    print("VERIFYING TRAINING DATA QUALITY")
    print("=" * 80)

    for idx in random.sample(range(len(train)), min(n_samples, len(train))):
        sample = train[idx]
        full = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

        user_count = full.count('<|start_header_id|>user<|end_header_id|>')
        assistant_count = full.count('<|start_header_id|>assistant<|end_header_id|>')
        if user_count != 1 or assistant_count != 1:
            print(f"\n⚠️  Sample {idx} has incorrect format!")
            print(f"   User tags: {user_count}, Assistant tags: {assistant_count}")
            print(f"   Text: {full[:200]}")

        labels = [t for t in sample['labels'] if t != -100]
        label_text = tokenizer.decode(labels, skip_special_tokens=True)
        if 'Translate to Portuguese' in label_text:
            print(f"\n⚠️  Sample {idx} has prompt in labels!")
            print(f"   Label text: {label_text[:100]}")

    print("Verification complete!")
    print("=" * 80 + "\n")


def main():
    # Initialize tracking BEFORE creating the trainer.
    weave.init(params.PROJECT_NAME)

    trainer = UniversalTrainer()
    model, tokenizer = trainer.get_model()

    print("[INFO] Loading datasets...")
    train, val, test = LanguageDS(
        tokenizer=tokenizer,
        dataset=params.DATASET,
    ).create_datasets(save=True)

    check_data_leakage(train, test, tokenizer)
    verify_data_quality(train, tokenizer)

    print(f"[INFO] Dataset sizes - Train: {len(train)}, "
          f"Val: {len(val) if val else 0}, OPUS Test: {len(test) if test else 0}")

    # Train (WandB is initialized inside train()).
    adapter_path = trainer.train(train, val)

    print("\n" + "=" * 80)
    print("TESTING WITH QUALITY FILTERING ENABLED")
    print("=" * 80)
    opus_results = trainer.test_generation(
        adapter_path=adapter_path,
        test_dataset=test,
        max_samples=None,
        use_quality_filter=True,
        verbose_filter=False,
        generation_strategy=params.DEFAULT_GENERATION_STRATEGY,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nWith Quality Filter:")
    print(f"  BLEU Score: {opus_results['metrics'].get('bleu', 0):.2f}")
    print(f"  Average Perplexity: {opus_results['avg_perplexity']:.2f}")
    print(f"  ROUGE-L F1: {opus_results['metrics'].get('rougeL_f1', 0):.4f}")

    if opus_results['filter_stats']:
        print("\nQuality Filter Impact:")
        print(f"  Pass Rate: {opus_results['filter_stats']['pass_rate'] * 100:.1f}%")
        print(f"  Repetitions Cleaned: {opus_results['filter_stats']['repetitions']}")
        print(f"  Language Mixing Fixed: {opus_results['filter_stats']['language_mixing']}")

    if params.USE_WANDB:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
