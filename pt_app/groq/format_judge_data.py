# format_judge_data.py
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split

import params

def format_for_qwen(evaluation_data):
    """
    Convert evaluation data to Qwen chat format
    """
    formatted_data = []
    
    for example in evaluation_data:
        # Qwen uses messages format
        messages = [
            {"role": "user", "content": example['input']},
            {"role": "assistant", "content": example['output']}
        ]
        
        formatted_data.append({
            "messages": messages,
            # Optional: keep original for reference
            "source": example.get('source', ''),
            "translation": example.get('translation', ''),
        })
    
    return formatted_data

def create_train_val_split(data, test_size=0.1, seed=42):
    """
    Split data into train and validation sets
    """
    train_data, val_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=seed
    )
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data

if __name__ == "__main__":
    # Load generated evaluation data
    print("Loading evaluation data...")
    with open(params.JUDGE_DATA_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Format for Qwen
    print("\nFormatting for Qwen training...")
    formatted_data = format_for_qwen(eval_data)
    
    # Split into train/val
    print("\nSplitting into train/validation...")
    train_data, val_data = create_train_val_split(formatted_data, test_size=0.1)
    
    # Save formatted datasets
    with open("judge_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("judge_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print("\n✓ Saved formatted datasets:")
    print(f"  - judge_train.json ({len(train_data)} examples)")
    print(f"  - judge_val.json ({len(val_data)} examples)")
    
    # Show example
    print("\n" + "="*60)
    print("EXAMPLE FORMATTED DATA:")
    print("="*60)
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))