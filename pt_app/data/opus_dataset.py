from datasets import load_dataset, concatenate_datasets
from mlx_lm_lora.trainer.datasets import TextDataset

from params import *


class OpusDS():
    
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # System prompt for Portuguese-English translation
        self.system_prompt = """You are a helpful assistant that translates Portuguese text to English. Provide accurate and natural translations."""

    def format_translation_prompt(self, sample):
        """
        Formats Portuguese-English translation pairs for instruction tuning.
        
        Expected input format:
        sample = {
            "portuguese": "Olá, como você está?",
            "english": "Hello, how are you?"
        }
        
        Or for exercise format:
        sample = {
            "source": "Olá, como você está?",
            "target": "Hello, how are you?",
            "source_lang": "pt",
            "target_lang": "en"
        }
        """
        
        # Handle different input formats
        if "portuguese" in sample and "english" in sample:
            portuguese_text = sample["portuguese"]
            english_text = sample["english"]
        elif "source" in sample and "target" in sample:
            if sample.get("source_lang") == "pt" and sample.get("target_lang") == "en":
                portuguese_text = sample["source"]
                english_text = sample["target"]
            elif sample.get("source_lang") == "en" and sample.get("target_lang") == "pt":
                # Reverse for Portuguese to English
                portuguese_text = sample["target"]
                english_text = sample["source"]
            else:
                portuguese_text = sample["source"]
                english_text = sample["target"]
        else:
            # Handle generic translation format
            portuguese_text = sample.get("text", sample.get("sentence", ""))
            english_text = sample.get("translation", sample.get("target", ""))
        
        # Create instruction-following format
        instruction = "Translate the following Portuguese text to English:"
        
        # Format as conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{portuguese_text}"},
            {"role": "assistant", "content": english_text}
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
        
        # Load OPUS-100 dataset (Portuguese-English subset)
        try:
            opus_dataset = load_dataset("opus_books", "en-pt")
            datasets.append(opus_dataset["train"])
            print(f"Loaded OPUS-100 pt-en: {len(opus_dataset['train'])} samples")
        except Exception as e:
            print(f"Could not load OPUS-100: {e}")
            
        return datasets
    
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
        
            # Filter out samples that are too long
        def filter_length(sample):
            return len(self.tokenizer.encode(sample["text"])) <= MAX_SEQ_LENGTH

        formatted_dataset = formatted_dataset.filter(filter_length)
        print(f"Samples after length filtering: {len(formatted_dataset)}")

        # Split into train and validation
        train_dataset, valid_dataset = formatted_dataset.train_test_split(
            test_size=0.1, 
            seed=42
        ).values()
        
        if save:
            train_dataset.save_to_disk("./datasets/opus/pt_en_train_dataset")
            valid_dataset.save_to_disk("./datasets/opus/pt_en_valid_dataset")
            
        
        train_set = TextDataset(train_dataset, self.tokenizer, text_key='text')
        valid_set = TextDataset(valid_dataset, self.tokenizer, text_key='text')
        
        return train_set, valid_set