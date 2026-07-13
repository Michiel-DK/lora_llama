"""
Model evaluation for the translation trainer.

Extracted from trainer_pt.py to keep the trainer focused on training. This module
runs comprehensive translation evaluation (BLEU/ROUGE/perplexity) with optional
quality filtering, and logs results to Weights & Biases when a run is active.

The public entry point is ``UniversalTrainer.test_generation`` in trainer_pt.py,
which delegates to ``run_evaluation`` here.
"""
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import wandb

import params
from pt_app.trainer.quality_filter import TranslationQualityFilter

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


def run_evaluation(
    trainer, 
    adapter_path=None, 
    test_dataset=None, 
    max_samples=None,
    use_quality_filter=True,
    verbose_filter=False,
    generation_strategy=None
):
    """
    Enhanced test translation with comprehensive metrics and quality filtering
    
    Args:
        adapter_path: Path to adapter weights
        test_dataset: Test dataset
        max_samples: Maximum samples to test
        use_quality_filter: Whether to apply quality filtering
        verbose_filter: Print filtering details for each sample
        generation_strategy: Generation strategy to use (greedy, beam_search, sampling)
                           If None, uses params.DEFAULT_GENERATION_STRATEGY
    """
    
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    print("[INFO] Using CPU for inference to avoid MPS memory limits")
    
    
    # Initialize scorers
    rouge_scorer_obj = None
    if ROUGE_AVAILABLE:
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    if adapter_path:
        # Reload model for inference
        base_model = AutoModelForCausalLM.from_pretrained(
            trainer.model_name,
            torch_dtype=torch.float32,
            cache_dir=params.CACHE_DIR,
            token=params.HF_TOKEN,
        )
        trainer.model = PeftModel.from_pretrained(base_model, adapter_path)
        trainer.model = trainer.model.to(trainer.device)
        
        # Reinitialize quality filter if not present
        if trainer.quality_filter is None:
            trainer.quality_filter = TranslationQualityFilter(
                tokenizer=trainer.tokenizer,
                target_language='pt'
            )
    else:
        # For baseline model, ensure it's on the correct device (CPU)
        trainer.model = trainer.model.to(trainer.device)
    
    trainer.model.eval()
    
    
    
    def extract_text_from_test_item(test_item):
        """Extract input and expected output from test dataset item"""
        input_ids = test_item['input_ids']
        labels = test_item['labels']
        
        # Find where labels start
        label_start_idx = None
        for i, label in enumerate(labels):
            if label != -100:
                label_start_idx = i
                break
        
        if label_start_idx is None:
            return None, None
        
        # Extract portions
        input_portion = input_ids[:label_start_idx]
        expected_labels = [label for label in labels if label != -100]
        
        # Decode
        input_text = trainer.tokenizer.decode(input_portion, skip_special_tokens=True)
        expected_output = trainer.tokenizer.decode(expected_labels, skip_special_tokens=True)
        
        # ============================================================
        # CLEAN UP DECODED TEXT
        # ============================================================
        # Remove role markers that slip through
        input_text = input_text.replace('user', '').strip()
        input_text = input_text.replace('assistant', '').strip()
        
        # Remove ellipsis artifacts
        input_text = input_text.replace('...', '').strip()
        
        # Remove any newlines
        input_text = input_text.replace('\n\n', ' ').replace('\n', ' ').strip()
        
        # Extract just the English text after instruction
        if 'Translate to Portuguese:' in input_text:
            parts = input_text.split('Translate to Portuguese:')
            if len(parts) > 1:
                input_text = parts[1].strip()
        
        # Clean expected output too
        expected_output = expected_output.replace('assistant', '').strip()
        expected_output = expected_output.replace('...', '').strip()
        # ============================================================
        
        return input_text, expected_output
    
    def calculate_perplexity(input_text, target_text):
        """Calculate perplexity for the target text"""
        full_text = input_text + target_text
        inputs = trainer.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
        
        prompt_inputs = trainer.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = trainer.model(**inputs, labels=inputs['input_ids'])
            shift_logits = outputs.logits[..., prompt_length-1:-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., prompt_length:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return torch.exp(loss).item()
    
    def calculate_metrics(predictions, references):
        """Calculate ROUGE and BLEU scores"""
        metrics = {}
        
        # ROUGE scores
        try:
            if ROUGE_AVAILABLE and rouge_scorer_obj:
                rouge_scores = defaultdict(list)
                for pred, ref in zip(predictions, references):
                    scores = rouge_scorer_obj.score(ref, pred)
                    for metric, score in scores.items():
                        rouge_scores[f"{metric}_f1"].append(score.fmeasure)
                
                for metric, scores in rouge_scores.items():
                    metrics[metric] = np.mean(scores)
        except Exception as e:
            print(f"⚠️  ROUGE calculation failed: {e}")
        
        # BLEU scores
        try:
            if BLEU_AVAILABLE:
                refs_formatted = [[ref] for ref in references]
                bleu_score = sacrebleu.corpus_bleu(predictions, refs_formatted)
                metrics['bleu'] = bleu_score.score
                metrics['bleu_precisions'] = bleu_score.precisions
        except Exception as e:
            print(f"⚠️  BLEU calculation failed: {e}")
        
        return metrics
    
    # Prepare test data
    if test_dataset is not None:
        # Use your actual test dataset
        if max_samples:
            test_subset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        else:
            test_subset = test_dataset
        
        prompts = []
        references = []
        
        print(f"Processing {len(test_subset)} test samples...")
        
    # Prepare test data
    if test_dataset is not None and len(test_dataset) > 0:
        print(f"✅ Using test dataset with {len(test_dataset)} samples")
        
        # Limit samples if requested
        if max_samples:
            test_subset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        else:
            test_subset = test_dataset
        
        prompts = []
        references = []
        
        print(f"Processing {len(test_subset)} test samples...")
        
        for i, test_item in enumerate(test_subset):
            # Extract clean English text
            if 'source_text' in test_item and test_item['source_text']:
                english_text = test_item['source_text']
                expected_output = test_item['target_text']
            else:
                english_text, expected_output = extract_text_from_test_item(test_item)
            
            if english_text is None or expected_output is None or not english_text.strip():
                print(f"⚠️  Skipping sample {i}: empty text")
                continue
            
            # ✅ BUILD PROPER PROMPT (matching your training format!)
            prompt = f"""<|start_header_id|>user<|end_header_id|>

    Translate to Portuguese: {english_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
            
            prompts.append(prompt)
            references.append(expected_output)
            
            # Show first 3 examples
            if i < 3:
                print(f"Example {i+1}:")
                print(f"  Input: {english_text[:100]}...")
                print(f"  Expected: {expected_output}")
        
        if len(prompts) == 0:
            print("⚠️  No valid samples extracted! Falling back to test sentences.")
            test_dataset = None  # Trigger fallback

    if test_dataset is None or len(prompts) == 0:
        # Fallback to simple test sentences
        print("⚠️  Using fallback test sentences")
        test_sentences = ["Hello!", "Thank you.", "Good morning."]
        prompts = []
        references = ["Olá!", "Obrigado.", "Bom dia."]
        
        for sentence in test_sentences:
            # ✅ Use YOUR custom format (not the system message one!)
            prompt = f"""<|start_header_id|>user<|end_header_id|>

    Translate to Portuguese: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
            prompts.append(prompt)
        
    # Set generation strategy
    if generation_strategy is None:
        generation_strategy = params.DEFAULT_GENERATION_STRATEGY
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TRANSLATION EVALUATION")
    if use_quality_filter:
        print("(WITH QUALITY FILTERING)")
    print(f"Generation Strategy: {generation_strategy.upper()}")
    print("="*80)
    
    # Generate predictions and calculate metrics
    raw_predictions = []
    filtered_predictions = []
    perplexities = []
    filtering_data = []
    
    for i, (prompt, reference) in enumerate(zip(prompts, references)):
        # Always show progress (not just when verbose_filter is True)
        if i % 10 == 0 or i < 5:
            print(f"[Progress] Processing sample {i+1}/{len(prompts)}...")
        
        if verbose_filter:
            print(f"\n{'='*60}")
            print(f"Processing sample {i+1}/{len(prompts)}")
            print(f"{'='*60}")
        
        # Generate translation with quality filtering
        raw_translation, filtered_translation, gen_config_used = trainer.generate_translation(
            prompt=prompt,
            generation_strategy=generation_strategy,
            use_quality_filter=use_quality_filter,
            verbose=verbose_filter
        )
        
        raw_predictions.append(raw_translation)
        
        # Use filtered version if available, otherwise use raw
        final_prediction = filtered_translation if filtered_translation else raw_translation
        filtered_predictions.append(final_prediction)
        
        # Track filtering stats
        filtering_data.append((
            prompt.split('user<|end_header_id|>')[-1].split('<|eot_id|>')[0].strip(),
            raw_translation,
            filtered_translation
        ))
        
        # Calculate perplexity on the final prediction
        try:
            perplexity = calculate_perplexity(prompt, final_prediction)
            perplexities.append(perplexity)
        except Exception as e:
            perplexities.append(float('inf'))
        
        # Show examples
        if i < 5:
            print(f"\nExample {i+1}:")
            if test_dataset is None:
                english_text = prompt.split('user<|end_header_id|>\n\n')[1].split('<|eot_id|>')[0].strip()
                print(f"  EN: {english_text}")
            else:
                print(f"  Input: {prompt[:100]}...")
            
            print(f"  Raw Output: {raw_translation}")
            if use_quality_filter and filtered_translation != raw_translation:
                print(f"  Filtered: {filtered_translation}")
                print(f"  [FILTERED]" if filtered_translation is None else "  [CLEANED]")
            print(f"  Expected: {reference}")
            print(f"  Perplexity: {perplexities[-1]:.2f}")
        
        # Clear memory on MPS
        if trainer.device_type == "mps" and i % 10 == 0:
            trainer._clear_memory()
    
    # Get filtering statistics
    if use_quality_filter and trainer.quality_filter:
        filter_stats = trainer.quality_filter.get_statistics(filtering_data)
        print(f"\n{'='*50}")
        print("QUALITY FILTER STATISTICS")
        print("="*50)
        print(f"Total samples: {filter_stats['total']}")
        print(f"Passed filter: {filter_stats['total'] - filter_stats['filtered_out']}")
        print(f"Filtered out: {filter_stats['filtered_out']}")
        print(f"Pass rate: {filter_stats['pass_rate']*100:.1f}%")
        print(f"\nFilter reasons:")
        print(f"  Language mixing: {filter_stats['language_mixing']}")
        print(f"  Repetitions cleaned: {filter_stats['repetitions']}")
        print(f"  Length issues: {filter_stats['length_issues']}")
        print(f"  Incomplete: {filter_stats['incomplete']}")
    
    # Calculate and display metrics
    print(f"\n{'='*50}")
    print("EVALUATION METRICS")
    print("="*50)
    
    print(f"Total samples: {len(filtered_predictions)}")
    print(f"Avg prediction length: {np.mean([len(p.split()) for p in filtered_predictions]):.2f} words")
    print(f"Avg reference length: {np.mean([len(r.split()) for r in references]):.2f} words")
    
    # Perplexity
    valid_perplexities = [p for p in perplexities if not math.isinf(p)]
    if valid_perplexities:
        print(f"\nPerplexity:")
        print(f"  Mean: {np.mean(valid_perplexities):.2f}")
        print(f"  Median: {np.median(valid_perplexities):.2f}")
        print(f"  Range: {np.min(valid_perplexities):.2f} - {np.max(valid_perplexities):.2f}")
    
    # ROUGE and BLEU (on filtered predictions)
    metrics = calculate_metrics(filtered_predictions, references)
    if metrics:
        print(f"\nROUGE Scores:")
        for key, value in metrics.items():
            if 'rouge' in key:
                print(f"  {key}: {value:.4f}")
        
        if 'bleu' in metrics:
            print(f"\nBLEU Score: {metrics['bleu']:.2f}")
            if 'bleu_precisions' in metrics:
                print(f"  Precisions: {[f'{p:.2f}' for p in metrics['bleu_precisions']]}")
    
    #### CHECK FOR BLEU
    
    # After calculating metrics, add this:
    if BLEU_AVAILABLE:
        try:
            print("\n" + "="*80)
            print("BLEU CALCULATION DEBUG")
            print("="*80)

            print("\nFirst 10 samples:")
            for i in range(min(10, len(filtered_predictions))):
                pred = filtered_predictions[i]
                ref = references[i]
                
                # Individual BLEU
                sent_bleu = sacrebleu.sentence_bleu(pred, [ref])
                
                print(f"\n{i+1}. Pred: {pred[:80]}...")
                print(f"   Ref:  {ref[:80]}...")
                print(f"   BLEU: {sent_bleu.score:.2f}")
                print(f"   Exact match: {'✅' if pred.strip() == ref.strip() else '❌'}")

            # Check if all predictions are somehow identical to references
            exact_matches = sum(1 for p, r in zip(filtered_predictions, references) if p.strip() == r.strip())
            print(f"\nExact matches: {exact_matches}/{len(filtered_predictions)} ({exact_matches/len(filtered_predictions)*100:.1f}%)")

            # Recalculate BLEU manually
            corpus_bleu = sacrebleu.corpus_bleu(filtered_predictions, [[r] for r in references])
            print(f"\nRecalculated corpus BLEU: {corpus_bleu.score:.2f}")
            print(f"Precisions: {corpus_bleu.precisions}")
            print("="*80)
        except Exception as e:
            print(f"⚠️  BLEU debug section failed: {e}")
    
    #######
    
    # Compare raw vs filtered metrics if filtering was used
    if use_quality_filter:
        raw_metrics = calculate_metrics(raw_predictions, references)
        if raw_metrics and metrics:
            print(f"\n{'='*50}")
            print("RAW vs FILTERED COMPARISON")
            print("="*50)
            if 'bleu' in raw_metrics and 'bleu' in metrics:
                improvement = metrics['bleu'] - raw_metrics['bleu']
                print(f"BLEU: {raw_metrics['bleu']:.2f} → {metrics['bleu']:.2f} ({improvement:+.2f})")
            if 'rougeL_f1' in raw_metrics and 'rougeL_f1' in metrics:
                improvement = metrics['rougeL_f1'] - raw_metrics['rougeL_f1']
                print(f"ROUGE-L: {raw_metrics['rougeL_f1']:.4f} → {metrics['rougeL_f1']:.4f} ({improvement:+.4f})")
    
    print("="*80)
    
    if wandb.run is not None:  # Check if wandb is initialized
            wandb.log({
                # Generation strategy
                "test/generation_strategy": generation_strategy,
                
                # Main metrics
                "test/bleu": metrics.get('bleu', 0),
                "test/rouge1_f1": metrics.get('rouge1_f1', 0),
                "test/rouge2_f1": metrics.get('rouge2_f1', 0),
                "test/rougeL_f1": metrics.get('rougeL_f1', 0),
                
                # Perplexity
                "test/perplexity_mean": np.mean(valid_perplexities) if valid_perplexities else float('inf'),
                "test/perplexity_median": np.median(valid_perplexities) if valid_perplexities else float('inf'),
                
                # Filter stats
                "test/filter_pass_rate": filter_stats['pass_rate'] if filter_stats else 1.0,
                "test/filter_repetitions": filter_stats['repetitions'] if filter_stats else 0,
                "test/filter_language_mixing": filter_stats['language_mixing'] if filter_stats else 0,
                "test/filter_incomplete": filter_stats['incomplete'] if filter_stats else 0,
                
                # Length stats
                "test/avg_prediction_length": np.mean([len(p.split()) for p in filtered_predictions]),
                "test/avg_reference_length": np.mean([len(r.split()) for r in references]),
            })
            
            # 2️⃣ Log example translations as a WandB Table
            examples_table = wandb.Table(
                columns=[
                    "ID", 
                    "Input (truncated)", 
                    "Raw Output", 
                    "Filtered Output",
                    "Reference", 
                    "Perplexity",
                    "Passed Filter"
                ],
                data=[
                    [
                        i,
                        prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i],
                        raw_predictions[i][:150] + "..." if len(raw_predictions[i]) > 150 else raw_predictions[i],
                        (filtered_predictions[i][:150] + "..." if filtered_predictions[i] and len(filtered_predictions[i]) > 150 else filtered_predictions[i]) or "FILTERED OUT",
                        references[i][:150] + "..." if len(references[i]) > 150 else references[i],
                        f"{perplexities[i]:.2f}" if not math.isinf(perplexities[i]) else "∞",
                        "✅" if filtered_predictions[i] else "❌"
                    ]
                    for i in range(min(20, len(prompts)))  # Log first 20 examples
                ]
            )
            wandb.log({"test/translation_examples": examples_table})
            
            # 3️⃣ Log comparison metrics if using filter
            if use_quality_filter and raw_metrics:
                wandb.log({
                    "test/raw_bleu": raw_metrics.get('bleu', 0),
                    "test/filtered_bleu": metrics.get('bleu', 0),
                    "test/bleu_improvement": metrics.get('bleu', 0) - raw_metrics.get('bleu', 0),
                    "test/raw_rouge_l": raw_metrics.get('rougeL_f1', 0),
                    "test/filtered_rouge_l": metrics.get('rougeL_f1', 0),
                })
        
    results = {
        'generation_strategy': generation_strategy,
        'generation_config': params.GENERATION_CONFIGS[generation_strategy],
        'raw_predictions': raw_predictions,
        'filtered_predictions': filtered_predictions,
        'references': references,
        'perplexities': perplexities,
        'metrics': metrics,
        'avg_perplexity': np.mean(valid_perplexities) if valid_perplexities else float('inf'),
        'filter_stats': filter_stats if use_quality_filter else None,
        
        # 4️⃣ Add structured examples for Weave
        'examples': [
            {
                'id': i,
                'input': prompts[i],
                'raw_output': raw_predictions[i],
                'filtered_output': filtered_predictions[i],
                'reference': references[i],
                'perplexity': perplexities[i],
                'passed_filter': filtered_predictions[i] is not None,
            }
            for i in range(len(prompts))
        ]
    }
    
    return results
