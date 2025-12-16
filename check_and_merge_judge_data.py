#!/usr/bin/env python3
"""
Check for duplicates across judge training files and merge them.
"""

import json
from pathlib import Path

def load_json(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save JSON data to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_fingerprint(example):
    """Create a unique fingerprint for an example based on input text."""
    # Use the input field as the unique identifier
    return example.get('input', '').strip()

def main():
    # Files to merge
    files = [
        'judge_training_data_merged2.json 10-49-59-838.json',
        'judge_low_training_data_cleaned.json',
        'judge_medium_data_progress_130.json'
    ]
    
    print("Loading files...")
    all_data = {}
    file_stats = {}
    
    for filepath in files:
        data = load_json(filepath)
        file_stats[filepath] = {
            'total': len(data),
            'unique': 0,
            'duplicates_within': 0,
            'duplicates_across': 0
        }
        
        print(f"\n{filepath}:")
        print(f"  Total examples: {len(data)}")
        
        seen_in_file = set()
        for example in data:
            fp = create_fingerprint(example)
            
            # Check for duplicates within this file
            if fp in seen_in_file:
                file_stats[filepath]['duplicates_within'] += 1
                continue
            seen_in_file.add(fp)
            
            # Check for duplicates across files
            if fp in all_data:
                file_stats[filepath]['duplicates_across'] += 1
                # Keep the first occurrence
                continue
            
            all_data[fp] = example
            file_stats[filepath]['unique'] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("DUPLICATE ANALYSIS:")
    print("="*60)
    
    total_examples = 0
    total_unique = 0
    
    for filepath, stats in file_stats.items():
        print(f"\n{filepath}:")
        print(f"  Total examples: {stats['total']}")
        print(f"  Unique to this file: {stats['unique']}")
        print(f"  Duplicates within file: {stats['duplicates_within']}")
        print(f"  Duplicates from previous files: {stats['duplicates_across']}")
        total_examples += stats['total']
        total_unique += stats['unique']
    
    print("\n" + "="*60)
    print(f"TOTAL: {total_examples} examples across all files")
    print(f"UNIQUE: {len(all_data)} examples after deduplication")
    print(f"REMOVED: {total_examples - len(all_data)} duplicates")
    print("="*60)
    
    # Convert back to list and analyze score distribution
    merged_data = list(all_data.values())
    
    print("\nScore distribution in merged dataset:")
    scores = {}
    for example in merged_data:
        # Extract score from output
        output = example.get('output', '')
        if 'Score:' in output:
            score_line = output.split('\n')[0]
            score = score_line.split(':')[1].strip().split('/')[0].strip()
            try:
                score_num = int(score)
                scores[score_num] = scores.get(score_num, 0) + 1
            except ValueError:
                pass
    
    for score in sorted(scores.keys()):
        print(f"  Score {score}: {scores[score]} examples")
    
    # Save merged data
    output_file = 'judge_training_data_merged_final.json'
    save_json(merged_data, output_file)
    print(f"\n✅ Merged data saved to: {output_file}")
    print(f"   Total examples: {len(merged_data)}")

if __name__ == '__main__':
    main()
