# dataset_handler.py
from typing import Dict, List, Optional
from datasets import Dataset
import json
import re

class TranslationDatasetProcessor:
    """Process translation dataset in Llama-3 chat format"""
    
    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Ensure tokenizer has proper chat template
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            print("[WARNING] Tokenizer missing chat template, using default Llama-3 template")
            self._set_llama3_chat_template()
    
    def _set_llama3_chat_template(self):
        """Set Llama-3.1 chat template if missing"""
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
        )
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset for training"""
        
        # The dataset already has the text in the correct format!
        # We just need to tokenize it properly
        
        def tokenize_function(examples):
            """Tokenize the text field"""
            # The text is already formatted with chat template
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # Don't pad here, let DataCollator handle it
                max_length=self.max_seq_length,
                return_overflowing_tokens=False,
            )
            
            # Copy input_ids to labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove original columns
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def mask_instructions_in_labels(self, dataset: Dataset) -> Dataset:
        """
        Optional: Mask system/user parts in labels so model only learns on assistant responses
        This can improve training quality for instruction following
        """
        
        def mask_labels(examples):
            """Mask everything except assistant responses"""
            for idx in range(len(examples["input_ids"])):
                input_ids = examples["input_ids"][idx]
                labels = examples["labels"][idx]
                
                # Find assistant response sections
                # Convert to string to find patterns
                text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                
                # Find all assistant sections
                assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
                
                # Create mask
                masked_labels = [-100] * len(labels)  # -100 is ignored in loss
                
                # Only unmask assistant responses
                current_pos = 0
                for match in re.finditer(assistant_pattern, text):
                    start_char = match.start(1)  # Start of actual response
                    end_char = match.end(1)      # End of actual response
                    
                    # Convert character positions to token positions (approximate)
                    prefix = text[:start_char]
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                    start_token = len(prefix_tokens)
                    
                    response_text = text[start_char:end_char]
                    response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
                    end_token = start_token + len(response_tokens)
                    
                    # Unmask assistant response tokens
                    for i in range(start_token, min(end_token, len(labels))):
                        masked_labels[i] = labels[i]
                
                examples["labels"][idx] = masked_labels
            
            return examples
        
        return dataset.map(
            mask_labels,
            batched=True,
            desc="Masking non-assistant tokens"
        )