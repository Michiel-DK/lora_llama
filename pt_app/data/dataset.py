# language_ds.py - ACTUAL FIX (removes default system prompt)
from datasets import load_dataset, Dataset
import os
import pandas as pd
from params import *
from tqdm import tqdm


class LanguageDS:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CRITICAL FIX: Override the chat template to remove default system message
        # This is the ACTUAL solution
        self._override_chat_template()
    
    def _override_chat_template(self):
        """
        Override Llama's chat template to remove the default system message
        that includes dates and knowledge cutoff
        """
        # Simplified chat template without the automatic system message
        custom_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        
        self.tokenizer.chat_template = custom_template
        print("[INFO] Custom chat template applied (removed default system message)")

    def process_sample(self, sample):
        """
        Process a single sample from OPUS or other datasets.
        Handles multiple dataset formats.
        """
        # Extract text based on dataset format
        if 'translation' in sample:
            english_text = sample['translation']['en']
            portuguese_text = sample['translation']['pt']
        elif 'english' in sample and 'portuguese' in sample:
            english_text = sample['english']
            portuguese_text = sample['portuguese']
        elif 'En' in sample and 'Pt' in sample:
            english_text = sample['En']
            portuguese_text = sample['Pt']
        elif 'English' in sample and 'Portuguese' in sample:
            english_text = sample['English']
            portuguese_text = sample['Portuguese']
        else:
            raise ValueError(f"Unknown format. Available keys: {sample.keys()}")
        
        # Skip empty translations
        if not english_text or not portuguese_text:
            return {"input_ids": None, "labels": None, "source_text": None, "target_text": None}
        
        # Skip very short texts
        MIN_WORDS = 5
        if (len(english_text.split()) < MIN_WORDS or 
            len(portuguese_text.split()) < MIN_WORDS):
            return {"input_ids": None, "labels": None, "source_text": None, "target_text": None}
        
        # Create conversation
        conversation = [
            {"role": "user", "content": f"Translate to Portuguese: {english_text}"},
            {"role": "assistant", "content": portuguese_text}
        ]
        
        # Tokenize prompt
        conversation_without_assistant = conversation[:-1]
        prompt_encoding = self.tokenizer.apply_chat_template(
            conversation_without_assistant,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            max_length=int(MAX_SEQ_LENGTH/2),
            return_tensors=None
        )
        
        # Check prompt length
        if len(prompt_encoding) > 200:
            return {"input_ids": None, "labels": None, "source_text": None, "target_text": None}
        
        # Tokenize full conversation
        full_encoding = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            truncation=False,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None
        )
        
        # Check total length
        MIN_TOTAL_TOKENS = 30
        MAX_TOTAL_TOKENS = 512
        
        if len(full_encoding) < MIN_TOTAL_TOKENS or len(full_encoding) > MAX_TOTAL_TOKENS:
            return {"input_ids": None, "labels": None, "source_text": None, "target_text": None}
        
        # Create labels
        prompt_length = len(prompt_encoding)
        labels = [-100] * prompt_length + full_encoding[prompt_length:]
        
        # ============================================================
        # ✅ ADD THIS - Store original text for testing!
        # ============================================================
        return {
            "input_ids": full_encoding,
            "labels": labels,
            "source_text": english_text,     
            "target_text": portuguese_text  
        }

    def create_datasets(self, save=True):
        """
        Create train/valid/test datasets from OPUS or Kaggle data.
        """
        print(f"Loading dataset: {self.dataset}")
        
        if self.dataset == 'opus_books':
            raw_dataset = load_dataset("opus_books", "en-pt")['train']
            print(f"Loaded OPUS books en-pt: {len(raw_dataset)} samples")
            
        elif self.dataset == 'opus100':
            raw_dataset = load_dataset("opus100", "en-pt")['train']
            print(f"Loaded OPUS-100 en-pt: {len(raw_dataset)} samples")
            
        elif self.dataset == 'kaggle':
            file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'por.txt')
            
            try:
                df = pd.read_csv(file, sep="\t", nrows=5)
                print(f"Detected columns: {df.columns.tolist()}")
                
                df = pd.read_csv(file, sep="\t")
                
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'eng' in col_lower or col in ['En', 'English', 'english']:
                        column_mapping[col] = 'English'
                    elif 'port' in col_lower or 'pt' in col_lower or col in ['Pt', 'Portuguese', 'portuguese']:
                        column_mapping[col] = 'Portuguese'
                
                df = df.rename(columns=column_mapping)
                df = df[['English', 'Portuguese']].dropna()
                
            except Exception as e:
                print(f"Error loading Kaggle dataset: {e}")
                df = pd.read_csv(
                    file, 
                    sep="\t", 
                    names=["En", "Pt", "NAN"],
                    usecols=["En", "Pt"]
                )
                df = df.rename(columns={"En": "English", "Pt": "Portuguese"})
            
            raw_dataset = Dataset.from_pandas(df)
            print(f"Loaded Kaggle dataset: {len(raw_dataset)} samples")
            print(f"Sample row: {raw_dataset[0]}")
            
        elif self.dataset == 'flores':
            flores = load_dataset(
                "facebook/flores",
                "eng_Latn-por_Latn",
                trust_remote_code=False,
                download_mode="force_redownload"  # Force clean download
            )

            
            def map_flores(ex):
                return {'translation': {'en': ex['sentence_eng_Latn'], 'pt': ex['sentence_por_Latn']}}
            
            # Get both splits
            val = flores['dev'].map(map_flores)
            test = flores['devtest'].map(map_flores)
            
            # Process both
            val_processed = [self.process_sample(s) for s in val if s]
            val_processed = [x for x in val_processed if x['input_ids']]
            
            test_processed = [self.process_sample(s) for s in test if s]
            test_processed = [x for x in test_processed if x['input_ids']]
            
            return None, Dataset.from_list(val_processed), Dataset.from_list(test_processed)
                    
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
            num_proc=None,
        )
        
        # Filter out None values
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
            print("SAMPLE DECODED TEXT:")
            print(sample_text)
            print("="*50)
            
            # Decode only the non-masked labels
            label_tokens = [t for t in sample['labels'] if t != -100]
            label_text = self.tokenizer.decode(label_tokens)
            print("\nLABEL TEXT (what model predicts):")
            print(label_text)
            print("="*50 + "\n")
            
            # DIAGNOSTIC: Show prompt length breakdown
            prompt_token_count = sum(1 for label in sample['labels'] if label == -100)
            label_token_count = sum(1 for label in sample['labels'] if label != -100)
            
            print(f"TOKEN BREAKDOWN:")
            print(f"  Prompt tokens: {prompt_token_count}")
            print(f"  Label tokens: {label_token_count}")
            print(f"  Total tokens: {len(sample['input_ids'])}")
            print(f"  Prompt %: {prompt_token_count / len(sample['input_ids']) * 100:.1f}%")
            print("="*50 + "\n")
        
        if save:
            save_dir = f"./datasets/{self.dataset}"
            os.makedirs(save_dir, exist_ok=True)
            train_dataset.save_to_disk(f"{save_dir}/train")
            valid_dataset.save_to_disk(f"{save_dir}/valid")
            test_dataset.save_to_disk(f"{save_dir}/test")
            print(f"Datasets saved to {save_dir}")
        
        return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    # print("="*80)
    # print("TESTING CUSTOM CHAT TEMPLATE (REMOVES DEFAULT SYSTEM)")
    # print("="*80)
    
    # ds = LanguageDS(tokenizer, dataset='opus100')
    # train, val, test = ds.create_datasets(save=False)
    
    # print("\nFirst 3 samples:")
    # for i in range(min(3, len(train))):
    #     text = tokenizer.decode(train[i]['input_ids'])
    #     print(f"\n{'='*60}")
    #     print(f"Sample {i+1}:")
    #     print(f"{'='*60}")
    #     print(text)
        
        
    import ipdb; ipdb.set_trace()
        
        
    ds = LanguageDS(tokenizer, dataset='opus100')
    train, val, test = ds.create_datasets(save=False)
    
    print("\nFirst 3 samples:")
    for i in range(min(3, len(train))):
        text = tokenizer.decode(train[i]['input_ids'])
        print(f"\n{'='*60}")
        print(f"Sample {i+1}:")
        print(f"{'='*60}")
        print(text)
        
    import ipdb; ipdb.set_trace()