# universal_trainer.py
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import params

from collections import defaultdict
import math

# Import the custom modules
from pt_app.trainer.quality_filter import TranslationQualityFilter
from pt_app.trainer.stopping_criteria import create_stopping_criteria_list

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge-score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")
    BLEU_AVAILABLE = False

import wandb
import weave



class UniversalTrainer:
    """Simple trainer that works on both MPS and CUDA with quality filtering"""
    
    def __init__(self):
        # Load configs from params
        self.model_name = params.MODEL_NAME
        self.adapter_path = params.ADAPTER_PATH
        self.max_seq_length = params.MAX_SEQ_LENGTH
        
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "mps"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        
        print(f"[INFO] Using device: {self.device_type}")
        
        # Device-specific settings
        self.batch_size = 8 if self.device_type == "cuda" else 2
        self.clear_memory_every = 50 if self.device_type == "cuda" else 5
        
        self.model = None
        self.tokenizer = None
        self.quality_filter = None
        
        os.makedirs(self.adapter_path, exist_ok=True)
        
    
    def get_model(self, apply_lora: bool = True) -> Tuple[Any, Any]:
        """
        Load model - works on both MPS and CUDA
        
        Args:
            apply_lora: If True, applies LoRA adapters. If False, loads base model only.
        """
        print(f"[INFO] Loading model: {self.model_name}")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Universal dtype
            trust_remote_code=True,
            cache_dir=params.CACHE_DIR,
            token=params.HF_TOKEN,
            use_cache=False,  # Disable for training
        )
        
        # Disable gradient checkpointing
        self.model.gradient_checkpointing_disable()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=params.CACHE_DIR,
            token=params.HF_TOKEN,
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize quality filter
        self.quality_filter = TranslationQualityFilter(
            tokenizer=self.tokenizer,
            target_language='pt'
        )
        print("[INFO] Quality filter initialized")
        
        # Apply LoRA (only if requested)
        if apply_lora:
            lora_config = LoraConfig(
                r=params.LORA_CONFIG["r"],
                lora_alpha=params.LORA_CONFIG["lora_alpha"],
                lora_dropout=params.LORA_CONFIG["lora_dropout"],
                target_modules=params.LORA_CONFIG["target_modules"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print("[INFO] LoRA adapters applied")
        else:
            print("[INFO] Base model loaded WITHOUT LoRA adapters")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        return self.model, self.tokenizer
    
    def train(self, train_dataset, val_dataset=None, epochs=None):
        """Simple manual training loop - works everywhere"""
        epochs = epochs or params.EPOCHS
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device_type == "cuda"),
            collate_fn=self._collate_fn
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=params.OPTIMIZER_CONFIG["learning_rate"],
            weight_decay=params.OPTIMIZER_CONFIG["weight_decay"]
        )
        
        # Training
        self.model.train()
        print(f"[INFO] Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_losses = []
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(progress):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward & backward
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                
                # Optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Track loss
                epoch_losses.append(loss.item())
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear memory periodically
                if (step + 1) % self.clear_memory_every == 0:
                    self._clear_memory()
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            })
            
            if val_dataset:
                val_loss = self._validate(val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
                wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                
                should_stop, best_path = self._check_early_stopping(val_loss, epoch)
                
                if should_stop:
                    print(f"Returning best model: {best_path}")
                    return best_path
                    
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.adapter_path, f"{timestamp}_final")
        self.model.save_pretrained(save_path)
        print(f"[INFO] Model saved to {save_path}")
        
        wandb.log({'model_saved_path': save_path})
        
        return save_path
    
    def _check_early_stopping(self, current_val_loss, epoch):
        """
        Check if early stopping should be triggered.
        Returns: (should_stop, best_model_path)
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.best_adapter_path = None
        
        patience = 3
        min_delta = 0.01
        
        if current_val_loss < (self.best_val_loss - min_delta):
            # Improvement
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.best_adapter_path = f"./adapters/{timestamp}_best_ep{epoch+1}"
            self.model.save_pretrained(self.best_adapter_path)
            
            print(f"✅ New best model! Val loss: {current_val_loss:.4f}")
            return False, self.best_adapter_path
        else:
            # No improvement
            self.patience_counter += 1
            print(f"⚠️  Patience: {self.patience_counter}/{patience}")
            
            if self.patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                return True, self.best_adapter_path
            
            return False, None
        
    def generate_translation(
        self, 
        prompt: str, 
        generation_strategy: str = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        use_quality_filter: bool = True,
        verbose: bool = False
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Generate translation with quality filtering and stopping criteria
        
        Args:
            prompt: Input prompt
            generation_strategy: Strategy name from params.GENERATION_CONFIGS (greedy, beam_search, sampling)
                               If None, uses params.DEFAULT_GENERATION_STRATEGY
            max_new_tokens: Override max_new_tokens from strategy (optional)
            temperature: Override temperature from strategy (optional)
            top_p: Override top_p from strategy (optional)
            use_quality_filter: Whether to apply quality filtering
            verbose: Print filtering details
            
        Returns:
            Tuple of (raw_translation, filtered_translation, generation_config_dict)
        """
        # Select generation strategy
        if generation_strategy is None:
            generation_strategy = params.DEFAULT_GENERATION_STRATEGY
        
        if generation_strategy not in params.GENERATION_CONFIGS:
            raise ValueError(f"Unknown strategy '{generation_strategy}'. Choose from: {list(params.GENERATION_CONFIGS.keys())}")
        
        # Get base config from params
        gen_config_dict = params.GENERATION_CONFIGS[generation_strategy].copy()
        
        # Apply overrides if provided
        if max_new_tokens is not None:
            gen_config_dict['max_new_tokens'] = max_new_tokens
        if temperature is not None:
            gen_config_dict['temperature'] = temperature
        if top_p is not None:
            gen_config_dict['top_p'] = top_p
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[1]
        
        # Create stopping criteria (only for greedy/sampling, not for beam search)
        # Beam search has its own early stopping mechanism
        stopping_criteria = None
        if generation_strategy != "beam_search":
            stopping_criteria = create_stopping_criteria_list(
                tokenizer=self.tokenizer,
                prompt_length=prompt_length,
                max_new_tokens=gen_config_dict['max_new_tokens'],
                prevent_repetition=True,
                prevent_language_switch=True,
                check_after_tokens=100
            )
        
        # Create generation config from dict (excluding 'strategy' key)
        gen_params = {k: v for k, v in gen_config_dict.items() if k != 'strategy'}
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_params
        )
        
        if verbose:
            print(f"🔍 Using generation strategy: {generation_strategy}")
            print(f"🔍 Config: {gen_config_dict}")
        
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria if stopping_criteria else None,
                use_cache=(self.device_type == "cuda"),
            )
        
        # Decode raw output
        raw_translation = self.tokenizer.decode(
            outputs[0][prompt_length:], 
            skip_special_tokens=True
        ).strip()
        
        # Apply quality filter if requested
        filtered_translation = None
        if use_quality_filter and self.quality_filter:
            # Extract source text from prompt (simple heuristic)
            source_text = prompt.split('user<|end_header_id|>')[-1].split('<|eot_id|>')[0].strip()
            filtered_translation = self.quality_filter.filter_translation(
                source=source_text,
                translation=raw_translation,
                verbose=verbose
            )
        
        return raw_translation, filtered_translation, gen_config_dict
    
    @weave.op()
    def test_generation(
        self, 
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
        
        self.device = torch.device("cpu")
        self.device_type = "cpu"
        print("[INFO] Using CPU for inference to avoid MPS memory limits")
        
        
        # Initialize scorers
        rouge_scorer_obj = None
        if ROUGE_AVAILABLE:
            rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if adapter_path:
            # Reload model for inference
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                cache_dir=params.CACHE_DIR,
                token=params.HF_TOKEN,
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model = self.model.to(self.device)
            
            # Reinitialize quality filter if not present
            if self.quality_filter is None:
                self.quality_filter = TranslationQualityFilter(
                    tokenizer=self.tokenizer,
                    target_language='pt'
                )
        
        self.model.eval()
        
        
        
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
            input_text = self.tokenizer.decode(input_portion, skip_special_tokens=True)
            expected_output = self.tokenizer.decode(expected_labels, skip_special_tokens=True)
            
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
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            prompt_inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                shift_logits = outputs.logits[..., prompt_length-1:-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., prompt_length:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                return torch.exp(loss).item()
        
        def calculate_metrics(predictions, references):
            """Calculate ROUGE and BLEU scores"""
            metrics = {}
            
            # ROUGE scores
            if ROUGE_AVAILABLE and rouge_scorer_obj:
                rouge_scores = defaultdict(list)
                for pred, ref in zip(predictions, references):
                    scores = rouge_scorer_obj.score(ref, pred)
                    for metric, score in scores.items():
                        rouge_scores[f"{metric}_f1"].append(score.fmeasure)
                
                for metric, scores in rouge_scores.items():
                    metrics[metric] = np.mean(scores)
            
            # BLEU scores
            if BLEU_AVAILABLE:
                refs_formatted = [[ref] for ref in references]
                bleu_score = sacrebleu.corpus_bleu(predictions, refs_formatted)
                metrics['bleu'] = bleu_score.score
                metrics['bleu_precisions'] = bleu_score.precisions
            
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
            raw_translation, filtered_translation, gen_config_used = self.generate_translation(
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
            if self.device_type == "mps" and i % 10 == 0:
                self._clear_memory()
        
        # Get filtering statistics
        if use_quality_filter and self.quality_filter:
            filter_stats = self.quality_filter.get_statistics(filtering_data)
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
        
    
    def _collate_fn(self, batch):
        """Collate function for pre-tokenized data"""
        if "input_ids" in batch[0]:
            # Already tokenized
            max_len = max(len(x["input_ids"]) for x in batch)
            
            input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
            input_ids.fill_(self.tokenizer.pad_token_id)
            
            for i, x in enumerate(batch):
                seq_len = len(x["input_ids"])
                input_ids[i, :seq_len] = torch.tensor(x["input_ids"])
            
            return {"input_ids": input_ids, "labels": input_ids.clone()}
        else:
            # Need tokenization
            texts = [x["text"] for x in batch]
            encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded["labels"] = encoded["input_ids"].clone()
            return encoded
    
    def _validate(self, val_dataset):
        """Quick validation"""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size*2, collate_fn=self._collate_fn)
        
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def _clear_memory(self):
        """Clear memory based on device"""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()


if __name__ == "__main__":
    from pt_app.data.dataset import LanguageDS
    
    
    # Initialize tracking BEFORE creating trainer
    weave.init(params.PROJECT_NAME)

    wandb.init(
        project=params.PROJECT_NAME,
        name=f"{params.MODEL_NAME}-{params.DATASET}-{params.DATASET_SAMPLES}-{params.EPOCHS}ep-{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags=["baseline"],
        config={
            "model": params.MODEL_NAME,
            "dataset": params.DATASET,
            'n_samples': params.DATASET_SAMPLES,
            "max_seq_length": params.MAX_SEQ_LENGTH,
            "max_new_tokens": params.MAX_NEW_TOKENS,
            "batch_size": params.BATCH_SIZE,  # Will be set by trainer
            "epochs": params.EPOCHS,
            "lora_r": params.LORA_CONFIG["r"],
            "lora_alpha": params.LORA_CONFIG["lora_alpha"],
        }
    )
    
    # Initialize trainer
    trainer = UniversalTrainer()
    
    # Load model
    model, tokenizer = trainer.get_model()
    
    # Load datasets (already tokenized)
    print("[INFO] Loading datasets...")
    train, val, test = LanguageDS(
        tokenizer=tokenizer,
        dataset=params.DATASET,
    ).create_datasets(save=True)
        
    print("\n" + "="*80)
    print("VERIFYING TRAINING DATA QUALITY")
    print("="*80)

    # Check 10 random samples
    import random
    sample_indices = random.sample(range(len(train)), min(10, len(train)))

    for idx in sample_indices:
        sample = train[idx]
        full = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        
        # Count role markers
        user_count = full.count('<|start_header_id|>user<|end_header_id|>')
        assistant_count = full.count('<|start_header_id|>assistant<|end_header_id|>')
        
        # Should be exactly 1 of each
        if user_count != 1 or assistant_count != 1:
            print(f"\n⚠️  Sample {idx} has incorrect format!")
            print(f"   User tags: {user_count}, Assistant tags: {assistant_count}")
            print(f"   Text: {full[:200]}")
        
        # Check labels don't include prompt
        labels = [t for t in sample['labels'] if t != -100]
        label_text = tokenizer.decode(labels, skip_special_tokens=True)
        
        if 'Translate to Portuguese' in label_text:
            print(f"\n⚠️  Sample {idx} has prompt in labels!")
            print(f"   Label text: {label_text[:100]}")

    print("Verification complete!")
    print("="*80 + "\n")

    print(f"[INFO] Dataset sizes - Train: {len(train)}, Val: {len(val) if val else 0}, OPUS Test: {len(test) if test else 0}")
    # Train
    adapter_path = trainer.train(train, val)
    
    # Test with quality filtering enabled (RECOMMENDED)
    print("\n" + "="*80)
    print("TESTING WITH QUALITY FILTERING ENABLED")
    print("="*80)
    opus_results = trainer.test_generation(
        adapter_path=adapter_path,
        test_dataset=test,
        max_samples=None,  #SET TO NONE FOR FULL TESTSET
        use_quality_filter=True,
        verbose_filter=False,  # Set to True to see filtering details
        generation_strategy=None  # Uses params.DEFAULT_GENERATION_STRATEGY ("greedy")
                                  # Options: "greedy", "beam_search", "sampling"
    )
    
    # Evaluate on FLORES
    # flores_loader = LanguageDS(tokenizer=tokenizer, dataset="flores")
    # _, flores_val, flores_test = flores_loader.create_datasets(save=True)

    # flores_results = trainer.test_generation(
    #     adapter_path=adapter_path,
    #     test_dataset=flores_test,  #SET TO NONE FOR FULL TESTSET
    #     max_samples=20,  #SET TO NONE FOR FULL TESTSET
    #     use_quality_filter=True,
    #     verbose_filter=False  # Set to True to see filtering details
    # )
    
    # # Log to WandB
    # wandb.log({
    #     "opus_bleu": opus_results['metrics']['bleu'],
    #     "flores_bleu": flores_results['metrics']['bleu'],
    #     "domain_transfer": flores_results['metrics']['bleu'] / opus_results['metrics']['bleu']
    # })
    
    # print(f"\n{'='*80}")
    # print(f"OPUS BLEU:   {opus_results['metrics']['bleu']:.2f}")
    # print(f"FLORES BLEU: {flores_results['metrics']['bleu']:.2f}")
    # print(f"Transfer:    {flores_results['metrics']['bleu'] / opus_results['metrics']['bleu'] * 100:.1f}%")
    # print(f"{'='*80}")
    
    
    wandb.finish()
    
    # Optionally test without filtering for comparison
    # print("\n" + "="*80)
    # print("TESTING WITHOUT QUALITY FILTERING (for comparison)")
    # print("="*80)
    # results_raw = trainer.test_generation(
    #     adapter_path=adapter_path,
    #     test_dataset=test,
    #     max_samples=30,
    #     use_quality_filter=False
    # )
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    # print("\nWithout Quality Filter:")
    # print(f"  BLEU Score: {results_raw['metrics'].get('bleu', 0):.2f}")
    # print(f"  Average Perplexity: {results_raw['avg_perplexity']:.2f}")
    # print(f"  ROUGE-L F1: {results_raw['metrics'].get('rougeL_f1', 0):.4f}")
    
    print("\nWith Quality Filter:")
    print(f"  BLEU Score: {opus_results['metrics'].get('bleu', 0):.2f}")
    print(f"  Average Perplexity: {opus_results['avg_perplexity']:.2f}")
    print(f"  ROUGE-L F1: {opus_results['metrics'].get('rougeL_f1', 0):.4f}")
    
    if opus_results['filter_stats']:
        print(f"\nQuality Filter Impact:")
        print(f"  Pass Rate: {opus_results['filter_stats']['pass_rate']*100:.1f}%")
        print(f"  Repetitions Cleaned: {opus_results['filter_stats']['repetitions']}")
        print(f"  Language Mixing Fixed: {opus_results['filter_stats']['language_mixing']}")
    
    # Calculate improvements
    # bleu_improvement = results_filtered['metrics'].get('bleu', 0) - results_raw['metrics'].get('bleu', 0)
    # rouge_improvement = results_filtered['metrics'].get('rougeL_f1', 0) - results_raw['metrics'].get('rougeL_f1', 0)
    
    # print(f"\nOverall Improvements:")
    # print(f"  BLEU: {bleu_improvement:+.2f} points")
    # print(f"  ROUGE-L: {rouge_improvement:+.4f} points")
    # print("="*80)