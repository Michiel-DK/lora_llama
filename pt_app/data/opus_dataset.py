# language_ds.py - FIXED VERSION
from datasets import load_dataset, Dataset
import os
import pandas as pd
from params import *


class LanguageDS:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.system_prompt = """You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations."""

    def process_sample(self, sample):
            """
            Process a single sample from OPUS or other datasets.
            Handles the actual OPUS format: {'translation': {'en': '...', 'pt': '...'}}
            """
            # Extract text based on dataset format
            if 'translation' in sample:
                # OPUS format
                english_text = sample['translation']['en']
                portuguese_text = sample['translation']['pt']
            elif 'english' in sample and 'portuguese' in sample:
                # Kaggle format
                english_text = sample['english']
                portuguese_text = sample['portuguese']
            elif 'En' in sample and 'Pt' in sample:
                # Raw kaggle format
                english_text = sample['En']
                portuguese_text = sample['Pt']
            else:
                raise ValueError(f"Unknown format: {sample.keys()}")
            
            # Skip empty translations
            if not english_text or not portuguese_text:
                return None
            
            # Format as conversation
            conversation = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": english_text},
                {"role": "assistant", "content": portuguese_text}
            ]
            
            # Apply chat template and tokenize - FIRST WITHOUT THE ASSISTANT RESPONSE
            conversation_without_assistant = conversation[:-1]  # Just system + user
            
            prompt_encoding = self.tokenizer.apply_chat_template(
                conversation_without_assistant,
                tokenize=True,
                add_generation_prompt=True,  # Adds the assistant header
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors=None
            )
            
            # NOW WITH THE FULL CONVERSATION
            full_encoding = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors=None
            )
            
            # CREATE LABELS: mask everything except the assistant's response
            prompt_length = len(prompt_encoding)
            labels = [-100] * prompt_length + full_encoding[prompt_length:]
            
            return {
                "input_ids": full_encoding,
                "labels": labels
            }

    def create_datasets(self, save=False):
        """
        Create train/valid/test datasets from OPUS or Kaggle data.
        """
        print(f"Loading dataset: {self.dataset}")
        
        if self.dataset == 'opus_books':
            # Load OPUS dataset
            raw_dataset = load_dataset("opus_books", "en-pt")['train']
            print(f"Loaded OPUS books en-pt: {len(raw_dataset)} samples")
            
        elif self.dataset == 'opus100':
            # Alternative: OPUS-100 which is cleaner
            raw_dataset = load_dataset("opus100", "en-pt")['train']
            print(f"Loaded OPUS-100 en-pt: {len(raw_dataset)} samples")
            
        elif self.dataset == 'kaggle':
            file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'por.txt')
            df = pd.read_csv(
                file, 
                sep="\t", 
                names=["En", "Pt", "NAN"],
                usecols=["En", "Pt"]
            )
            raw_dataset = Dataset.from_pandas(df)
            print(f"Loaded Kaggle dataset: {len(raw_dataset)} samples")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # Limit samples if specified
        if DATASET_SAMPLES and DATASET_SAMPLES < len(raw_dataset):
            raw_dataset = raw_dataset.select(range(DATASET_SAMPLES))
            print(f"Limited to {DATASET_SAMPLES} samples")
        
        # Process all samples
        processed_dataset = raw_dataset.map(
            self.process_sample,
            remove_columns=raw_dataset.column_names,
            desc="Processing translations",
            num_proc=None,  # Single process for MPS compatibility
        )
        
        # Filter out None values (empty translations)
        processed_dataset = processed_dataset.filter(
            lambda x: x['input_ids'] is not None and x['labels'] is not None
        )
        
        print(f"Processed dataset size: {len(processed_dataset)}")
        
        # Calculate stats
        lengths = [len(ids) for ids in processed_dataset['input_ids']]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        print(f"Average token length: {avg_length:.2f}")
        print(f"Max token length: {max(lengths) if lengths else 0}")
        print(f"Min token length: {min(lengths) if lengths else 0}")
        
        # Split into train/valid/test (80/10/10)
        train_valid = processed_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_valid["train"]
        valid_test = train_valid["test"].train_test_split(test_size=0.5, seed=42)
        valid_dataset = valid_test["train"]
        test_dataset = valid_test["test"]
        
        print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(valid_dataset)}, Test: {len(test_dataset)}")
        
        # Verify format by decoding a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            sample_text = self.tokenizer.decode(sample['input_ids'])
            print("\n" + "="*50)
            print("SAMPLE DECODED TEXT (first 500 chars):")
            print(sample_text[:500])
            print("="*50)
            
            # Decode only the non-masked labels to show what the model will predict
            label_tokens = [t for t in sample['labels'] if t != -100]
            label_text = self.tokenizer.decode(label_tokens)
            print("\nLABEL TEXT (what model predicts):")
            print(label_text[:500])
            print("="*50 + "\n")
        
        if save:
            save_dir = f"./datasets/{self.dataset}"
            os.makedirs(save_dir, exist_ok=True)
            train_dataset.save_to_disk(f"{save_dir}/train")
            valid_dataset.save_to_disk(f"{save_dir}/valid")
            test_dataset.save_to_disk(f"{save_dir}/test")
            print(f"Datasets saved to {save_dir}")
        
        return train_dataset, valid_dataset, test_dataset

# Test the dataset class
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # Create dataset
    ds = LanguageDS(tokenizer, dataset='opus100')  # or 'opus_books'
    train, val, test = ds.create_datasets(save=True)
    
    # Check a few samples
    print("\nFirst 3 training samples decoded:")
    for i in range(min(3, len(train))):
        text = tokenizer.decode(train[i]['input_ids'])
        print(f"\nSample {i+1}:")
        print(text[:1000] + "...")