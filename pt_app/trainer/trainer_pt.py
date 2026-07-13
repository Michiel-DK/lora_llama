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
        self.run_timestamp = params.RUN_TIMESTAMP
        self.run_name = None  
        
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
    
    def train(self, train_dataset, val_dataset=None, epochs=None, save_name=None):
        """Simple manual training loop - works everywhere"""
        epochs = epochs or params.EPOCHS
        
        # Create run name with timestamp
        dataset_name = params.DATASET
        if save_name:
            self.run_name = f"{save_name}_{self.run_timestamp}"
        else:
            self.run_name = f"{params.MODEL_NAME.split('/')[-1]}-{dataset_name}-{epochs}ep-{self.run_timestamp}"
        
        # ✅ ADDED: Initialize WandB with matching name
        if params.USE_WANDB:
            wandb.init(
                project=params.WANDB_PROJECT,
                name=self.run_name,  # ← SAME as adapter save name!
                tags=["training"],
                config={
                    # Model configuration
                    "model/name": params.MODEL_NAME,
                    "model/lora_r": params.LORA_CONFIG["r"],
                    "model/lora_alpha": params.LORA_CONFIG["lora_alpha"],
                    "model/lora_dropout": params.LORA_CONFIG["lora_dropout"],
                    
                    # Dataset configuration
                    "dataset/name": dataset_name,
                    "dataset/samples_requested": params.DATASET_SAMPLES,
                    "dataset/samples_actual": len(train_dataset),  # ← Actual samples after filtering
                    "dataset/min_words": params.MIN_WORDS,
                    "dataset/min_total_tokens": params.MIN_TOTAL_TOKENS,
                    "dataset/max_total_tokens": params.MAX_TOTAL_TOKENS,
                    
                    # Training configuration
                    "training/epochs": epochs,
                    "training/batch_size": self.batch_size,
                    "training/learning_rate": params.OPTIMIZER_CONFIG["learning_rate"],
                    "training/weight_decay": params.OPTIMIZER_CONFIG["weight_decay"],
                    "training/gradient_accumulation_steps": params.GRADIENT_ACCUMULATION_STEPS,
                    
                    # Metadata
                    "timestamp": self.run_timestamp,
                }
            )
        
        print(f"\n{'='*80}")
        print(f"TRAINING: {self.run_name}")
        print(f"{'='*80}\n")
        
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
            
            if params.USE_WANDB:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                })
            
            if val_dataset:
                val_loss = self._validate(val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
                
                if params.USE_WANDB:
                    wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                
                should_stop, best_path = self._check_early_stopping(val_loss, epoch)
                
                if should_stop:
                    print(f"🛑 Early stopping triggered!")
                    print(f"Returning best model: {best_path}")
                    
                    if params.USE_WANDB:
                        wandb.log({"final_model_path": best_path})
                    
                    return best_path
        
        # ✅ CHANGED: Save final model with consistent naming
        final_adapter_path = f"{self.adapter_path}{self.run_name}_final"
        self.model.save_pretrained(final_adapter_path)
        print(f"[INFO] Model saved to {final_adapter_path}")
        
        if params.USE_WANDB:
            wandb.log({'model_saved_path': final_adapter_path})
            
            # Save model as artifact
            artifact = wandb.Artifact(
                name=f"adapter-{self.run_name}",
                type="model",
                description=f"LoRA adapter trained on {params.DATASET}"
            )
            artifact.add_dir(final_adapter_path)
            wandb.log_artifact(artifact)
        
        return final_adapter_path
    
    def _check_early_stopping(self, current_val_loss, epoch):
        """
        Check if early stopping should be triggered.
        Returns: (should_stop, best_model_path)
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.best_adapter_path = None
        
        patience = getattr(params, 'EARLY_STOPPING_PATIENCE', 3)
        min_delta = getattr(params, 'EARLY_STOPPING_MIN_DELTA', 0.01)
        
        if current_val_loss < (self.best_val_loss - min_delta):
            # Improvement
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            
            # ✅ CHANGED: Save best model with consistent naming
            self.best_adapter_path = f"{self.adapter_path}{self.run_name}_best_ep{epoch+1}"
            self.model.save_pretrained(self.best_adapter_path)
            
            print(f"✅ New best model! Val loss: {current_val_loss:.4f}")
            
            if params.USE_WANDB:
                wandb.log({
                    "best_val_loss": current_val_loss,
                    "best_epoch": epoch + 1,
                    "best_model_path": self.best_adapter_path
                })
            
            return False, self.best_adapter_path
        else:
            # No improvement
            self.patience_counter += 1
            print(f"⚠️  Patience: {self.patience_counter}/{patience}")
            
            if self.patience_counter >= patience:
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
        """Run comprehensive translation evaluation. See pt_app.trainer.evaluation."""
        from pt_app.trainer.evaluation import run_evaluation
        return run_evaluation(
            self,
            adapter_path=adapter_path,
            test_dataset=test_dataset,
            max_samples=max_samples,
            use_quality_filter=use_quality_filter,
            verbose_filter=verbose_filter,
            generation_strategy=generation_strategy,
        )

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