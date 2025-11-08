# hf_trainer_final_fix.py
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import params


class HFTrainerFinal:
    """FINAL WORKING trainer for Llama models on MPS"""
    
    def __init__(self):
        self.model_name = params.MODEL_NAME
        self.adapter_path = params.ADAPTER_PATH
        self.output_dir = params.OUTPUT_DIR
        self.cache_dir = params.CACHE_DIR
        
        self.batch_size = params.BATCH_SIZE
        self.epochs = params.EPOCHS
        self.max_seq_length = params.MAX_SEQ_LENGTH
        
        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
        # Create directories
        os.makedirs(self.adapter_path, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_model(self) -> Tuple[Any, Any]:
        """Load model with WORKING configuration for MPS"""
        
        print(f"[INFO] Loading model: {self.model_name}")
        
        # CRITICAL FIX 1: Load in float32 for MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # MPS needs float32
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=params.HF_TOKEN,
            use_cache=False,  # CRITICAL: Disable cache
            low_cpu_mem_usage=True,
        )
        
        # CRITICAL FIX 2: Explicitly disable gradient checkpointing
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = False
        self.model.config.gradient_checkpointing = False
        
        # For each layer, ensure gradient checkpointing is off
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                for layer in self.model.model.layers:
                    if hasattr(layer, 'gradient_checkpointing'):
                        layer.gradient_checkpointing = False
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=params.HF_TOKEN,
        )
        
        # CRITICAL FIX 3: Proper padding for Llama
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Special pad token
            self.tokenizer.pad_token_id = 128004  # Llama-3 specific
        
        # CRITICAL FIX 4: Simple LoRA config that works
        lora_config = LoraConfig(
            r=16,  # Increase rank for better learning
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # All attention
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # CRITICAL FIX 5: Ensure model is in training mode
        self.model.train()
        
        # Move to MPS
        self.model = self.model.to(self.device)
        
        # Print stats
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, dataset):
        """Universal dataset preparation for all model sizes"""
        
        # Works for 1B, 3B, 8B+ models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        def tokenize_function(examples):
            # Same tokenization for all sizes
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Universal best practice
                max_length=self.max_seq_length,
                return_tensors=None,
            )
            
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        # Disable multiprocessing for MPS (all sizes)
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=None,  # Universal for MPS
        )
        
        return tokenized
    
    def train(self, train_dataset, val_dataset=None):
        """Training with WORKING configuration"""
        
        # Prepare datasets
        print("[INFO] Preparing datasets...")
        # train_dataset = self.prepare_dataset(train_dataset)
        # if val_dataset:
        #     val_dataset = self.prepare_dataset(val_dataset)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        
        # CRITICAL: Working training arguments for MPS
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,  # Small batch for MPS
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # Accumulate to effective batch of 4
            warmup_steps=20,
            learning_rate=3e-4,  # Good for LoRA
            weight_decay=0.001,
            logging_steps=1,
            save_strategy="steps",
            save_steps=20,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=20 if val_dataset else None,
            save_total_limit=2,
            load_best_model_at_end=False,  # Disable to avoid issues
            metric_for_best_model="loss",
            greater_is_better=False,
            
            # CRITICAL MPS settings
            fp16=False,  # MPS doesn't support fp16 training
            bf16=False,
            gradient_checkpointing=False,  # MUST BE FALSE
            gradient_checkpointing_kwargs=None,
            
            # Optimizer settings
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            
            # Other settings
            report_to="none",
            run_name=f"translation_{timestamp}",
            disable_tqdm=False,
            seed=42,
            data_seed=42,
            
            # MPS specific
            use_mps_device=True,
            dataloader_num_workers=0,  # Must be 0 for MPS
            dataloader_pin_memory=False,  # Must be False for MPS
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,  # Don't pad additionally
            return_tensors="pt",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # CRITICAL: Ensure gradient checkpointing is OFF before training
        trainer.model.gradient_checkpointing_disable()
        
        print(f"[INFO] Starting training...")
        print(f"[INFO] Train samples: {len(train_dataset)}")
        print(f"[INFO] Epochs: {self.epochs}")
        print(f"[INFO] Steps per epoch: {len(train_dataset) // (1 * 4)}")  # batch_size * grad_accum
        
        # Train
        train_result = trainer.train()
        
        # Save adapter
        final_adapter_path = os.path.join(self.adapter_path, f"{timestamp}_final")
        trainer.save_model(final_adapter_path)
        
        print(f"[INFO] Training complete!")
        print(f"[INFO] Final loss: {train_result.training_loss:.4f}")
        print(f"[INFO] Adapter saved to: {final_adapter_path}")
        
        return final_adapter_path
    
    def test_generation(self, adapter_path=None):
        """Test translation with proper generation"""
        
        if adapter_path:
            print(f"[INFO] Loading adapter: {adapter_path}")
            # Reload base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                cache_dir=self.cache_dir,
                token=params.HF_TOKEN,
                use_cache=True,  # Enable cache for inference
            )
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to eval mode
        
        test_sentences = [
            "Hello, how are you?",
            "Good morning!",
            "Thank you.",
        ]
        
        print("\n" + "="*60)
        print("TRANSLATION TESTS")
        print("="*60)
        
        for sentence in test_sentences:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.device)
            
            # Generate with working settings for MPS
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=30,
                    min_new_tokens=2,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode only the generated part
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            print(f"\nEN: {sentence}")
            print(f"PT: {response}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Your actual dataset loading
    from pt_app.data.opus_dataset import LanguageDS
    
    # Initialize trainer
    trainer = HFTrainerFinal()
    
    # Load model
    model, tokenizer = trainer.get_model()
    
    # Load datasets
    print("[INFO] Loading datasets...")
    train, val, test = LanguageDS(
        tokenizer=tokenizer,
        dataset='opus_books'
    ).create_datasets(save=True)
    
    import ipdb; ipdb.set_trace()
    
    # Limit for testing if needed
    if params.DATASET_SAMPLES:
        train = train.select(range(min(params.DATASET_SAMPLES, len(train))))
    
    print(f"[INFO] Training on {len(train)} samples")
    
    # Train
    adapter_path = trainer.train(train, val)
    
    # Test generation
    trainer.test_generation(adapter_path)