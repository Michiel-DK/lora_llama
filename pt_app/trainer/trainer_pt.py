# universal_trainer.py
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import params


class UniversalTrainer:
    """Simple trainer that works on both MPS and CUDA"""
    
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
        
        os.makedirs(self.adapter_path, exist_ok=True)
    
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
            
            # Validate if provided
            if val_dataset and epoch == epochs - 1:  # Only validate at end
                val_loss = self._validate(val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.adapter_path, f"{timestamp}_final")
        self.model.save_pretrained(save_path)
        print(f"[INFO] Model saved to {save_path}")
        
        return save_path
    
    def test_generation(self, adapter_path=None):
        """Test translation - works on both devices"""
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
        
        self.model.eval()
        
        test_sentences = ["Hello!", "Thank you.", "Good morning."]
        
        print("\n" + "="*50)
        print("TRANSLATION TESTS")
        print("="*50)
        
        for sentence in test_sentences:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                    You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations in Portuguese. <|eot_id|><|start_header_id|>user<|end_header_id|>

                    {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                    """
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with device-appropriate settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.3,
                    do_sample=True,
                    use_cache=(self.device_type == "cuda"),  # Only use cache on CUDA
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            import ipdb;ipdb.set_trace()
            
            print(f"EN: {sentence} -> PT: {response}")
            
            # Clear memory on MPS
            if self.device_type == "mps":
                self._clear_memory()
        
        print("="*50)
    
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
    
    import ipdb;ipdb.set_trace()
    
    # # Add labels if needed
    # if "labels" not in train[0]:
    #     train = train.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    #     if val:
    #         val = val.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    print(f"[INFO] Dataset sizes - Train: {len(train)}, Val: {len(val) if val else 0}")
    
    # Train
    adapter_path = trainer.train(train, val)
    
    # Test
    trainer.test_generation(adapter_path)