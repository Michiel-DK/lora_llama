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
        
        ##wandb initialization
        wandb.init(
        project="translation-llama",
        config={
            "model": self.model_name,
            "device": self.device_type,
            "dataset": params.DATASET,
            "max_seq_length": self.max_seq_length,
            "max_new_tokens": params.MAX_NEW_TOKENS,
            'batch_size': self.batch_size,
            'epochs': params.EPOCHS,
            "lora_r": params.LORA_CONFIG["r"],
            "lora_alpha": params.LORA_CONFIG["lora_alpha"],
        }
    )
    
    def get_model(self) -> Tuple[Any, Any]:
        """Load model - works on both MPS and CUDA"""
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
        
        # Apply LoRA
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
            
            # Validate if provided
            if val_dataset and epoch == epochs - 1:  # Only validate at end
                val_loss = self._validate(val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
                wandb.log({"val_loss": val_loss})
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.adapter_path, f"{timestamp}_final")
        self.model.save_pretrained(save_path)
        print(f"[INFO] Model saved to {save_path}")
        
        wandb.log({'model_saved_path': save_path})
        
        return save_path
    
    def generate_translation(
        self, 
        prompt: str, 
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_quality_filter: bool = True,
        verbose: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Generate translation with quality filtering and stopping criteria
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_quality_filter: Whether to apply quality filtering
            verbose: Print filtering details
            
        Returns:
            Tuple of (raw_translation, filtered_translation)
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[1]
        
        # Create stopping criteria
        stopping_criteria = create_stopping_criteria_list(
            tokenizer=self.tokenizer,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prevent_repetition=True,
            prevent_language_switch=True,
            check_after_tokens=100
        )
        
        # Create generation config with improved parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,        # Prevent 3-word repetitions
            repetition_penalty=1.2,         # Penalize repeated tokens
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print(f"🔍 DEBUG: generation_config.max_new_tokens = {generation_config.max_new_tokens}")
    
        # ... before generate ...
        print(f"🔍 DEBUG: About to generate with input shape {inputs['input_ids'].shape}")
        
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
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
        
        return raw_translation, filtered_translation
    
    def test_generation(
        self, 
        adapter_path=None, 
        test_dataset=None, 
        max_samples=None,
        use_quality_filter=True,
        verbose_filter=False
    ):
        """
        Enhanced test translation with comprehensive metrics and quality filtering
        
        Args:
            adapter_path: Path to adapter weights
            test_dataset: Test dataset
            max_samples: Maximum samples to test
            use_quality_filter: Whether to apply quality filtering
            verbose_filter: Print filtering details for each sample
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
            
            # Find where labels start (not -100)
            label_start_idx = None
            for i, label in enumerate(labels):
                if label != -100:
                    label_start_idx = i
                    break
            
            if label_start_idx is None:
                return None, None
            
            # Extract input portion
            input_portion = input_ids[:label_start_idx]
            # Extract expected output
            expected_labels = [label for label in labels if label != -100]
            
            # Decode texts
            input_text = self.tokenizer.decode(input_portion, skip_special_tokens=True)
            expected_output = self.tokenizer.decode(expected_labels, skip_special_tokens=True)
            
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
            
            for i, test_item in enumerate(test_subset):
                input_text, expected_output = extract_text_from_test_item(test_item)
                
                if input_text is None or expected_output is None:
                    continue
                
                prompts.append(input_text)
                references.append(expected_output)
                
                if i < 3:  # Show first 3 examples
                    print(f"Example {i+1}:")
                    print(f"  Input: {input_text[:100]}...")
                    print(f"  Expected: {expected_output}")
            
        else:
            # Fallback to simple test sentences
            test_sentences = ["Hello!", "Thank you.", "Good morning."]
            prompts = []
            references = ["Olá!", "Obrigado.", "Bom dia."]
            
            for sentence in test_sentences:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                        You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations.<|eot_id|><|start_header_id|>user<|end_header_id|>

                        {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                        """
                prompts.append(prompt)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TRANSLATION EVALUATION")
        if use_quality_filter:
            print("(WITH QUALITY FILTERING)")
        print("="*80)
        
        # Generate predictions and calculate metrics
        raw_predictions = []
        filtered_predictions = []
        perplexities = []
        filtering_data = []
        
        for i, (prompt, reference) in enumerate(zip(prompts, references)):
            if verbose_filter:
                print(f"\n{'='*60}")
                print(f"Processing sample {i+1}/{len(prompts)}")
                print(f"{'='*60}")
            
            # Generate translation with quality filtering
            raw_translation, filtered_translation = self.generate_translation(
                prompt=prompt,
                max_new_tokens=params.MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.9,
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
        
        # Return results for programmatic use
        return {
            'raw_predictions': raw_predictions,
            'filtered_predictions': filtered_predictions,
            'references': references,
            'perplexities': perplexities,
            'metrics': metrics,
            'avg_perplexity': np.mean(valid_perplexities) if valid_perplexities else float('inf'),
            'filter_stats': filter_stats if use_quality_filter else None
        }
    
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
    from pt_app.data.opus_dataset import LanguageDS
    
    # Initialize trainer
    trainer = UniversalTrainer()
    
    # Load model
    model, tokenizer = trainer.get_model()
    
    # Load datasets (already tokenized)
    print("[INFO] Loading datasets...")
    train, val, test = LanguageDS(
        tokenizer=tokenizer,
        dataset='opus_books'
    ).create_datasets(save=True)
    
    print(f"[INFO] Dataset sizes - Train: {len(train)}, Val: {len(val) if val else 0}")
    
    # Train
    adapter_path = trainer.train(train, val)
    
    # Test with quality filtering enabled (RECOMMENDED)
    print("\n" + "="*80)
    print("TESTING WITH QUALITY FILTERING ENABLED")
    print("="*80)
    results_filtered = trainer.test_generation(
        adapter_path=adapter_path,
        test_dataset=test,
        max_samples=30,
        use_quality_filter=True,
        verbose_filter=False  # Set to True to see filtering details
    )
    
    wandb.log({
            "test_bleu": results_filtered['metrics'].get('bleu', 0),
            "test_rouge_l": results_filtered['metrics'].get('rougeL_f1', 0),
            "test_perplexity": results_filtered['avg_perplexity'],
            "filter_pass_rate": results_filtered['filter_stats']['pass_rate'] if results_filtered['filter_stats'] else 0,
        })

    # Optionally test without filtering for comparison
    print("\n" + "="*80)
    print("TESTING WITHOUT QUALITY FILTERING (for comparison)")
    print("="*80)
    results_raw = trainer.test_generation(
        adapter_path=adapter_path,
        test_dataset=test,
        max_samples=30,
        use_quality_filter=False
    )
    
    # Summary comparison
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nWithout Quality Filter:")
    print(f"  BLEU Score: {results_raw['metrics'].get('bleu', 0):.2f}")
    print(f"  Average Perplexity: {results_raw['avg_perplexity']:.2f}")
    print(f"  ROUGE-L F1: {results_raw['metrics'].get('rougeL_f1', 0):.4f}")
    
    print("\nWith Quality Filter:")
    print(f"  BLEU Score: {results_filtered['metrics'].get('bleu', 0):.2f}")
    print(f"  Average Perplexity: {results_filtered['avg_perplexity']:.2f}")
    print(f"  ROUGE-L F1: {results_filtered['metrics'].get('rougeL_f1', 0):.4f}")
    
    if results_filtered['filter_stats']:
        print(f"\nQuality Filter Impact:")
        print(f"  Pass Rate: {results_filtered['filter_stats']['pass_rate']*100:.1f}%")
        print(f"  Repetitions Cleaned: {results_filtered['filter_stats']['repetitions']}")
        print(f"  Language Mixing Fixed: {results_filtered['filter_stats']['language_mixing']}")
    
    # Calculate improvements
    bleu_improvement = results_filtered['metrics'].get('bleu', 0) - results_raw['metrics'].get('bleu', 0)
    rouge_improvement = results_filtered['metrics'].get('rougeL_f1', 0) - results_raw['metrics'].get('rougeL_f1', 0)
    
    print(f"\nOverall Improvements:")
    print(f"  BLEU: {bleu_improvement:+.2f} points")
    print(f"  ROUGE-L: {rouge_improvement:+.4f} points")
    print("="*80)