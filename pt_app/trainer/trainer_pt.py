# trainer_pt.py - Complete integrated training pipeline
import os
import torch
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

import params
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training,
    PeftModel
)
from torch.utils.data import DataLoader
import torch.nn.functional as F


class HFTrainer:
    """HuggingFace Trainer using params.py configuration"""
    
    def __init__(self):
        # Import all configs from params
        self.model_name = params.MODEL_NAME
        self.adapter_path = params.ADAPTER_PATH
        self.output_dir = params.OUTPUT_DIR
        self.cache_dir = params.CACHE_DIR
        
        # Training configs
        self.batch_size = params.BATCH_SIZE
        self.epochs = params.EPOCHS
        self.max_seq_length = params.MAX_SEQ_LENGTH
        
        # Configs
        self.lora_config = params.LORA_CONFIG
        self.optimizer_config = params.OPTIMIZER_CONFIG
        self.training_args = params.TRAINING_ARGS
        self.quantization_config = params.QUANTIZATION_CONFIG
        
        # Device setup
        self.device = self._setup_device()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Create directories
        os.makedirs(self.adapter_path, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _setup_device(self):
        """Setup compute device with proper detection"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Using CUDA: {torch.cuda.get_device_name()}")
            print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[INFO] Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            print("[WARNING] Using CPU - this will be slow!")
        return device
    
    def get_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer with quantization and LoRA"""
        
        print(f"[INFO] Loading model: {self.model_name}")
        
        # Setup quantization (only for CUDA)
        bnb_config = None
        if self.quantization_config["load_in_4bit"] and self.device.type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.quantization_config["bnb_4bit_compute_dtype"]),
                bnb_4bit_quant_type=self.quantization_config["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=self.quantization_config.get("bnb_4bit_use_double_quant", True),
            )
            print("[INFO] Using 4-bit quantization (BitsAndBytes)")
        
        # Determine dtype
        if self.device.type == "cuda":
            torch_dtype = torch.float16
        elif self.device.type == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device.type == "cuda" else None,
                dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                token=params.HF_TOKEN if params.HF_TOKEN else None,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("[INFO] Trying without authentication...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device.type == "cuda" else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=params.HF_TOKEN if params.HF_TOKEN else None,
        )
        
        # Setup padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        # Prepare for k-bit training if using quantization
        if bnb_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.training_args.get("gradient_checkpointing", True)
            )
        elif self.training_args.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias=self.lora_config["bias"],
            task_type=getattr(TaskType, self.lora_config["task_type"]),
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        # Move to device if not using auto device map
        if self.device.type == "mps":
            self.model = self.model.to(self.device)
        
        return self.model, self.tokenizer
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer using params configuration"""
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_config["learning_rate"],
            betas=(
                self.optimizer_config["adam_beta1"],
                self.optimizer_config["adam_beta2"]
            ),
            eps=self.optimizer_config["adam_epsilon"],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        
        print(f"[INFO] Optimizer configured with lr={self.optimizer_config['learning_rate']}")
        return self.optimizer
    
    def train(self, train_dataset, val_dataset=None):
        """Train the model using HuggingFace Trainer"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        
        # Update training arguments with device-specific settings
        training_args_dict = self.training_args.copy()
        training_args_dict["output_dir"] = output_dir
        
        # Device-specific settings
        if self.device.type == "cuda":
            training_args_dict["fp16"] = True
        elif self.device.type == "mps":
            training_args_dict["fp16"] = False
        else:
            training_args_dict["fp16"] = False
        
        # Remove any MLX-specific parameters
        training_args_dict.pop("use_mps_device", None)
        
        # Create training arguments
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        # Train
        print(f"[INFO] Starting training...")
        print(f"[INFO] Output directory: {output_dir}")
        
        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
            trainer.save_model(os.path.join(output_dir, "interrupted_checkpoint"))
            return output_dir
        
        # Save final model
        final_adapter_path = os.path.join(
            self.adapter_path, 
            f"{timestamp}_final"
        )
        trainer.save_model(final_adapter_path)
        print(f"[INFO] Model saved to {final_adapter_path}")
        
        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
            "timestamp": timestamp,
            "model_name": self.model_name,
            "effective_batch_size": self.batch_size * self.training_args["gradient_accumulation_steps"],
            "device": str(self.device),
        }
        
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return final_adapter_path
    
    def test(self, 
             test_dataset,
             adapter_path: Optional[str] = None,
             batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Test the model on a test dataset"""
        
        batch_size = batch_size or self.batch_size
        
        if adapter_path and adapter_path != self.adapter_path:
            print(f"[INFO] Loading adapter from {adapter_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device.type == "cuda" else None,
                torch_dtype=torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32,
                cache_dir=self.cache_dir,
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            if self.device.type == "mps":
                self.model = self.model.to(self.device)
        
        # Create data loader
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=0,
        )
        
        # Evaluation
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        print(f"[INFO] Testing on {len(test_dataset)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += (shift_labels != -100).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results = {
            "test_loss": avg_loss,
            "test_perplexity": perplexity,
            "total_samples": len(test_dataset),
            "total_tokens": total_tokens,
            "model": self.model_name,
            "adapter": adapter_path,
            "device": str(self.device),
        }
        
        print(f"\n{'='*50}")
        print(f"TEST RESULTS")
        print(f"{'='*50}")
        print(f"Loss:       {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Samples:    {len(test_dataset)}")
        print(f"{'='*50}\n")
        
        if adapter_path:
            results_path = os.path.join(
                os.path.dirname(adapter_path),
                "test_results.json"
            )
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[INFO] Results saved to {results_path}")
        
        return results
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n{'='*50}")
        print(f"TRAINABLE PARAMETERS")
        print(f"{'='*50}")
        print(f"Trainable: {trainable_params:,}")
        print(f"Total:     {all_param:,}")
        print(f"Ratio:     {100 * trainable_params / all_param:.2f}%")
        print(f"{'='*50}\n")


# Main execution
if __name__ == "__main__":
    from pt_app.data.opus_dataset import LanguageDS
    
    print("="*60)
    print("PORTUGUESE-ENGLISH TRANSLATION MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    print("\n[STEP 1] Initializing trainer...")
    trainer = HFTrainer()
    
    # Get model and tokenizer
    print("\n[STEP 2] Loading model and tokenizer...")
    model, tokenizer = trainer.get_model()
    optimizer = trainer.get_optimizer()
    
    # Load datasets
    print("\n[STEP 3] Loading and processing datasets...")
    language_ds = LanguageDS(
        tokenizer=tokenizer, 
        dataset='opus_books'  # or 'kaggle'
    )
    train_dataset, val_dataset, test_dataset = language_ds.create_datasets(save=True)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Train the model
    print("\n[STEP 4] Training model...")
    adapter_path = trainer.train(train_dataset, val_dataset)
    
    # Test the trained model
    print("\n[STEP 5] Testing model...")
    if test_dataset:
        test_results = trainer.test(
            test_dataset=test_dataset,
            adapter_path=adapter_path
        )
    
    # Generate sample translation
    print("\n[STEP 6] Testing translation capability...")
    test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(trainer.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\nGenerated translation:\n{response}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)