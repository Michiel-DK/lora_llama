from datasets import load_dataset, concatenate_datasets, Dataset
import os
from mlx_lm_lora.trainer.datasets import TextDataset

from params import *

import pandas as pd

import random


class LanguageDS():
    
    def __init__(self, tokenizer, dataset):
        
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # System prompt for English-Portuguese translation
        self.system_prompt = """You are a helpful assistant that translates English text to Portuguese. Provide accurate and natural translations."""

    def format_translation_prompt(self, sample):
        """
        Formats English-Portuguese translation pairs for instruction tuning.
        
        Expected input format:
        sample = {
            "english": "Hello, how are you?",
            "portuguese": "Olá, como você está?"
        }
        
        Or for exercise format:
        sample = {
            "source": "Hello, how are you?",
            "target": "Olá, como você está?",
            "source_lang": "en",
            "target_lang": "pt"
        }
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
                # Reverse for English to Portuguese
                english_text = sample["target"]
                portuguese_text = sample["source"]
            else:
                english_text = sample["source"]
                portuguese_text = sample["target"]
        else:
            # Handle generic translation format
            english_text = sample.get("text", sample.get("sentence", ""))
            portuguese_text = sample.get("translation", sample.get("target", ""))
        
        # Keep just the instruction without redundant prompt
        #instruction = "Translate the following English text to Portuguese:"
        
        # Format as conversation with proper assistant response
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{english_text}"},
            {"role": "assistant", "content": portuguese_text}  # Just the Portuguese text
        ]
        
        # Apply chat template
        sample["text"] = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return sample
    
    def create_translation_exercises_dataset(self):
        """
        Create a dataset from various Portuguese-English translation sources
        """
        datasets = []
        
        # Load OPUS-100 dataset (English-Portuguese subset)
        if self.dataset == 'opus_books':
            opus_dataset = load_dataset("opus_books", "en-pt")
            datasets.append(opus_dataset["train"])
            print(f"Loaded OPUS-100 en-pt: {len(opus_dataset['train'])} samples")
            
        elif self.dataset == 'kaggle':
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'por.txt')
            df = pd.read_csv(file_path, sep="\t", names=["En", "Pt", "NAN"])[["En", "Pt"]]
            # Convert to OPUS structure so existing extraction code works
            opus_format_data = []
            for _, row in df.iterrows():
                opus_format_data.append({
                    "translation": {
                        "en": row["En"],
                        "pt": row["Pt"]
                    }
                })
            
            # Convert to HuggingFace Dataset
            kaggle_dataset = Dataset.from_list(opus_format_data)
            datasets.append(kaggle_dataset)
            print(f"Loaded Kaggle Portuguese dataset: {len(kaggle_dataset)} samples")

        return datasets
    
    def print_formatted_samples(self, dataset, num_samples=3):
        """Print sample items from formatted dataset"""
        print("\n===== FORMATTED DATASET SAMPLES =====")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        for i, idx in enumerate(indices):
            print(f"Sample {i+1}:")
            print(dataset[idx]['text'])
            print('-' * 50)

    def create_datasets(self, save=False):
        
        datasets = self.create_translation_exercises_dataset()
        
        if datasets:
            combined_dataset = concatenate_datasets(datasets)
            
        if DATASET_SAMPLES is not None:
            combined_dataset = combined_dataset.select(range(min(DATASET_SAMPLES, len(combined_dataset))))
            
        formatted_dataset = combined_dataset.map(
            self.format_translation_prompt,
            remove_columns=combined_dataset.column_names,
            desc="Formatting translations"
            )
        
        self.print_formatted_samples(formatted_dataset)
        
        # Truncate samples that are too long instead of filtering
        def truncate_text(sample):
            tokens = self.tokenizer.encode(sample["text"])
            if len(tokens) > MAX_SEQ_LENGTH:
                # Truncate to MAX_SEQ_LENGTH tokens
                truncated_tokens = tokens[:MAX_SEQ_LENGTH]
                # Decode back to text
                sample["text"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return sample
        
        formatted_dataset = formatted_dataset.map(
            truncate_text,
            desc="Truncating long texts"
        )

        # Split into train and validation
        train_dataset, valid_dataset = formatted_dataset.train_test_split(
            test_size=0.1, 
            seed=42
        ).values()
        
        if save:
            train_dataset.save_to_disk(f"./datasets/{self.dataset}/pt_en_train_dataset")
            valid_dataset.save_to_disk(f"./datasets/{self.dataset}/pt_en_valid_dataset")


        train_set = TextDataset(train_dataset, self.tokenizer, text_key='text')
        valid_set = TextDataset(valid_dataset, self.tokenizer, text_key='text')
        
        return train_set, valid_set
        
if __name__ == '__main__':
    
    from pt_app.trainer.trainer import LloraTrainer
    
    lora = LloraTrainer()
    
    m, t = lora.get_model()
    
    ds_opus = LanguageDS(t, dataset='opus_books')
    train_opus, valid_opus = ds_opus.create_datasets(save=True)

    ds_kaggle = LanguageDS(t, dataset='kaggle')
    train_kaggle, valid_kaggle = ds_kaggle.create_datasets(save=True)

    import ipdb;ipdb.set_trace()