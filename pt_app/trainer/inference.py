"""
Translation Inference Engine

Lightweight class for running inference with trained LoRA adapters.
Optimized for production use - no training dependencies required.
"""

import os
import torch
from typing import Dict, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel

import params
from pt_app.trainer.quality_filter import TranslationQualityFilter
from pt_app.trainer.stopping_criteria import create_stopping_criteria_list


class TranslationInference:
    """
    Lightweight inference engine for English → Portuguese translation.
    
    Features:
    - Load trained LoRA adapters
    - Single prompt translation
    - Batch translation
    - Interactive mode
    - Quality filtering
    - Smart device selection (CPU/CUDA)
    
    Device Auto-Detection:
        - CUDA: Always used if available (fastest)
        - MPS: Disabled by default (crashes with model+adapter due to 4GB limit)
        - CPU: Default for Mac (stable, ~1-2s per prompt)
    
    MPS Note:
        Even small models (1B) crash on MPS when loaded with adapters because
        the combined memory exceeds the 4GB tensor allocation limit. You can
        try MPS with use_cpu=False but expect crashes.
    
    Example:
        >>> translator = TranslationInference()  # Auto-detects (CPU on Mac)
        >>> translator.load_adapter("./adapters/20251118_124144_best_ep1")
        >>> result = translator.translate("Hello, how are you?")
        >>> print(result['filtered'])
    """
    
    def __init__(self, model_name: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the inference engine.
        
        Args:
            model_name: Base model name (defaults to params.MODEL_NAME)
            force_cpu: Force CPU usage (ignores auto-detection)
        """
        self.model_name = model_name or params.MODEL_NAME
        
        # Auto-detect device - MPS disabled by default due to 4GB limit
        if force_cpu:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            print(f"[INFO] Forcing CPU for inference (user requested)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
            print(f"[INFO] Using CUDA for inference")
        elif torch.backends.mps.is_available():
            # MPS crashes even with 1B models due to 4GB tensor limit
            # when loading model + adapter together
            print(f"[INFO] MPS available but defaulting to CPU for stability")
            print(f"[INFO] (MPS has 4GB tensor limit that causes crashes with adapters)")
            print(f"[INFO] To try MPS anyway, pass use_cpu=False to load_adapter()")
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
            print(f"[INFO] Using CPU for inference")
        
        self.model = None
        self.tokenizer = None
        self.quality_filter = None
    
    def load_adapter(self, adapter_path: str, use_cpu: Optional[bool] = None):
        """
        Load a trained adapter for inference.
        
        Args:
            adapter_path: Path to the saved adapter (e.g., "./adapters/20251118_124144_best_ep1")
            use_cpu: Force CPU usage (None=auto-detect, True=force CPU, False=use device from init)
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"[INFO] Loading adapter for inference: {adapter_path}")
        
        # Handle device override
        if use_cpu is True:
            # User explicitly wants CPU
            if self.device_type != "cpu":
                print(f"[INFO] Overriding {self.device_type.upper()} to use CPU")
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        elif use_cpu is False:
            # User explicitly doesn't want CPU override - keep current device
            print(f"[INFO] Using {self.device_type.upper()} (CPU override disabled)")
        # else use_cpu is None: keep whatever was set in __init__
        
        # Load tokenizer
        print(f"[INFO] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=params.CACHE_DIR,
            token=params.HF_TOKEN,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print(f"[INFO] Loading base model: {self.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            cache_dir=params.CACHE_DIR,
            token=params.HF_TOKEN,
        )
        
        # Load adapter weights
        print(f"[INFO] Loading adapter weights...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize quality filter
        self.quality_filter = TranslationQualityFilter(
            tokenizer=self.tokenizer,
            target_language='pt'
        )
        print("[INFO] Quality filter initialized")
        
        print(f"[SUCCESS] Model ready for inference on {self.device_type}")
        return self.model, self.tokenizer
    
    def translate(
        self, 
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_quality_filter: bool = True,
        verbose: bool = False
    ) -> Dict[str, str]:
        """
        Translate a single English prompt to Portuguese.
        
        Args:
            prompt: Input text to translate (e.g., "Hello, how are you?")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            use_quality_filter: Apply post-processing quality filter
            verbose: Print detailed filtering information
        
        Returns:
            Dictionary with 'raw' and 'filtered' translations
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_adapter() first.")
        
        # Format the prompt with the chat template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Translate to Portuguese: {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate translation
        raw_translation, filtered_translation = self._generate_translation(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_quality_filter=use_quality_filter,
            verbose=verbose
        )
        
        return {
            'raw': raw_translation,
            'filtered': filtered_translation if filtered_translation else raw_translation
        }
    
    def batch_translate(
        self,
        prompts: list,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        use_quality_filter: bool = True,
        show_progress: bool = True
    ) -> list:
        """
        Translate multiple prompts in batch.
        
        Args:
            prompts: List of English texts to translate
            max_new_tokens: Maximum tokens per translation
            temperature: Sampling temperature
            use_quality_filter: Apply quality filtering
            show_progress: Show progress bar
        
        Returns:
            List of translation dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_adapter() first.")
        
        results = []
        iterator = enumerate(prompts, 1)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(prompts), desc="Translating")
            except ImportError:
                pass  # Fall back to simple enumeration
        
        for i, prompt in iterator:
            result = self.translate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_quality_filter=use_quality_filter,
                verbose=False
            )
            results.append(result)
            
            # Clear memory periodically on MPS
            if self.device_type == "mps" and i % 10 == 0:
                self._clear_memory()
        
        return results
    
    def interactive(
        self,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        use_quality_filter: bool = True
    ):
        """
        Start an interactive translation session.
        Type your prompts and get translations in real-time.
        Type 'quit' or 'exit' to stop.
        
        Args:
            max_new_tokens: Maximum tokens to generate per translation
            temperature: Sampling temperature
            use_quality_filter: Apply quality filtering
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_adapter() first.")
        
        print("\n" + "="*80)
        print("INTERACTIVE TRANSLATION MODE")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device_type}")
        print(f"Quality Filter: {'Enabled' if use_quality_filter else 'Disabled'}")
        print("\nType your English text and press Enter to get Portuguese translation.")
        print("Type 'quit' or 'exit' to stop.\n")
        print("="*80 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("English text: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                if not user_input:
                    print("Please enter some text.\n")
                    continue
                
                # Run inference
                print("\nTranslating...", end=" ", flush=True)
                result = self.translate(
                    prompt=user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_quality_filter=use_quality_filter,
                    verbose=False
                )
                
                # Display result
                print("\r" + " "*50 + "\r", end="")  # Clear "Translating..." message
                print(f"Portuguese: {result['filtered']}\n")
                
                if use_quality_filter and result['raw'] != result['filtered']:
                    print(f"[Note: Quality filter was applied]\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                continue
    
    def _generate_translation(
        self, 
        prompt: str, 
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_quality_filter: bool = True,
        verbose: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Internal method to generate translation with quality filtering.
        
        Args:
            prompt: Formatted input prompt
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
        
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
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
            # Extract source text from prompt
            source_text = prompt.split('user<|end_header_id|>')[-1].split('<|eot_id|>')[0].strip()
            if 'Translate to Portuguese:' in source_text:
                source_text = source_text.split('Translate to Portuguese:')[-1].strip()
            
            filtered_translation = self.quality_filter.filter_translation(
                source=source_text,
                translation=raw_translation,
                verbose=verbose
            )
        
        return raw_translation, filtered_translation
    
    def _clear_memory(self):
        """Clear memory based on device"""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <adapter_path> [prompt]")
        print("Example: python inference.py ./adapters/20251118_124144_best_ep1 'Hello!'")
        sys.exit(1)
    
    adapter_path = sys.argv[1]
    
    translator = TranslationInference()
    translator.load_adapter(adapter_path, use_cpu=True)
    
    if len(sys.argv) > 2:
        # Single prompt mode
        prompt = " ".join(sys.argv[2:])
        result = translator.translate(prompt)
        print(f"\nEnglish:    {prompt}")
        print(f"Portuguese: {result['filtered']}")
    else:
        # Interactive mode
        translator.interactive()
