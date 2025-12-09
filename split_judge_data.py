#!/usr/bin/env python3
"""
Split judge_training_data_final.json into train/val/test sets with messages format
"""
import json
import random
from pathlib import Path

def convert_to_messages_format(item):
    """Convert input/output format to messages format"""
    return {
        "messages": [
            {
                "role": "user",
                "content": item["input"]
            },
            {
                "role": "assistant", 
                "content": item["output"]
            }
        ],
        "source": "",
        "translation": ""
    }

def split_dataset(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test and convert to messages format
    
    Args:
        input_file: Path to judge_training_data_final.json
        output_dir: Directory to save split files
        train_ratio: Proportion for training (default 0.8 = 80%)
        val_ratio: Proportion for validation (default 0.1 = 10%)
        test_ratio: Proportion for test (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    # Load data
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split sizes
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    # Test gets the remainder to ensure we use all data
    
    # Split
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    # Convert to messages format
    print("\nConverting to messages format...")
    train_messages = [convert_to_messages_format(item) for item in train_data]
    val_messages = [convert_to_messages_format(item) for item in val_data]
    test_messages = [convert_to_messages_format(item) for item in test_data]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    print("\nSaving files...")
    
    files = {
        'judge_train.json': train_messages,
        'judge_val.json': val_messages,
        'judge_test.json': test_messages
    }
    
    for filename, data_to_save in files.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {output_path} ({len(data_to_save)} examples)")
    
    print("\n✅ Done! Files ready for training.")
    print(f"\nTo upload to VM:")
    print(f"scp -P <port> {output_dir}/*.json root@<ip>:/root/lora_llama/datasets/judge_eval/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split judge training data into train/val/test")
    parser.add_argument(
        "--input",
        default="datasets/judge_eval/judge_training_data_final.json",
        help="Input file (default: datasets/judge_eval/judge_training_data_final.json)"
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/judge_eval",
        help="Output directory (default: datasets/judge_eval)"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios sum to ~1.0
    total_ratio = args.train + args.val + args.test
    if not (0.99 <= total_ratio <= 1.01):
        print(f"⚠️  Warning: Ratios sum to {total_ratio:.2f}, not 1.0")
        print("   Continuing anyway, test set will get remainder...")
    
    split_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
