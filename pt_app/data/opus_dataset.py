from datasets import load_dataset, concatenate_datasets, Dataset
import os
import pandas as pd
from params import *


class LanguageDS():
    
    def __init__(self, tokenizer, dataset):
        
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_prompt = """You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations."""

    def extract_translation_pair(self, sample):
        """
        Extract English and Portuguese text from various formats.
        Handles: {"english": ..., "portuguese": ...}, {"source": ..., "target": ...}, etc.
        """
        # Handle different input formats
        if "english" in sample and "portuguese" in sample:
            english_text = sample["english"]
            portuguese_text = sample["portuguese"]
        elif "source" in sample and "target" in sample:
            if sample.get("source_lang") == "en" and sample.get("target_lang") == "pt":
                english_text = sample["source"]
                portuguese_text = sample["target"]
            elif sample.get("source_lang") == "pt" and sample.get("target_lang") == "en":
                english_text = sample["target"]
                portuguese_text = sample["source"]
            else:
                english_text = sample["source"]
                portuguese_text = sample["target"]
        else:
            english_text = sample.get("text", sample.get("sentence", ""))
            portuguese_text = sample.get("translation", sample.get("target", ""))
        
        return english_text, portuguese_text

    def process_sample(self, sample):
        """
        Combined operation: extract translation pair, format as conversation, 
        apply chat template, tokenize, and truncate.
        Returns properly formatted input_ids for HuggingFace trainer.
        """
        english_text, portuguese_text = self.extract_translation_pair(sample)
        
        # Format as conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": english_text},
            {"role": "assistant", "content": portuguese_text}
        ]
        
        # Apply chat template and tokenize in one step with truncation
        encoding = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None  # Return as list, not tensor
        )
        
        # Return only input_ids - DataCollator will handle attention_mask
        return {"input_ids": encoding}

    def create_translation_exercises_dataset(self):
        """
        Create a dataset from various Portuguese-English translation sources
        """
        datasets = []
        
        if self.dataset == 'opus_books':
            opus_dataset = load_dataset("opus_books", "en-pt")
            datasets.append(opus_dataset["train"])
            print(f"Loaded OPUS-100 en-pt: {len(opus_dataset['train'])} samples")
            
        elif self.dataset == 'kaggle':
            file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'por.txt')
            df = pd.read_csv(
                file, 
                sep="\t", 
                names=["En", "Pt", "NAN"],
                usecols=["En", "Pt"]
            )
            df = df.rename(columns={"En": "english", "Pt": "portuguese"})
            datasets.append(Dataset.from_pandas(df))

        return datasets
    
    def create_datasets(self, save=False):
        """
        Create train/valid/test datasets with optimized parallel processing.
        """
        datasets = self.create_translation_exercises_dataset()
        
        if not datasets:
            raise ValueError("No datasets created")
                
        combined_dataset = concatenate_datasets(datasets)
        
        # Limit samples if specified
        if DATASET_SAMPLES is not None:
            combined_dataset = combined_dataset.select(
                range(min(DATASET_SAMPLES, len(combined_dataset)))
            )
        
        # Single map operation with parallelization
        # Processes formatting, tokenization, and truncation in one pass
        processed_dataset = combined_dataset.map(
            self.process_sample,
            remove_columns=combined_dataset.column_names,
            desc="Processing translations (format + tokenize + truncate)",
            num_proc=4,  # Parallel processing with 4 workers
            batch_size=1000  # Process in batches for efficiency
        )
        
        print(f"Processed dataset size: {len(processed_dataset)}")
        lengths = [len(ids) for ids in processed_dataset['input_ids']]
        avg_length = sum(lengths) / len(lengths)
        print(f"Average token length: {avg_length:.2f}")
        
        # Split into train/valid/test
        train_valid = processed_dataset.train_test_split(
            test_size=0.1, 
            seed=42
        )
        train_dataset = train_valid["train"]
        valid_dataset = train_valid["test"]
        
        train_test = train_dataset.train_test_split(
            test_size=0.1, 
            seed=42
        )
        train_dataset = train_test["train"]
        test_dataset = train_test["test"]

        if save:
            train_dataset.save_to_disk("./datasets/opus/pt_en_train_dataset")
            valid_dataset.save_to_disk("./datasets/opus/pt_en_valid_dataset")
            test_dataset.save_to_disk("./datasets/opus/pt_en_test_dataset")
            print("Datasets saved to disk")
        
        return train_dataset, valid_dataset, test_dataset