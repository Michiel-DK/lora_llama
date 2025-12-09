#!/usr/bin/env python3
"""Check if final and splits contain different data"""
import json

# Load all files
with open('datasets/judge_eval/judge_training_data_final.json') as f:
    final_data = json.load(f)

with open('datasets/judge_eval/judge_train.json') as f:
    train_data = json.load(f)
    
with open('datasets/judge_eval/judge_val.json') as f:
    val_data = json.load(f)
    
with open('datasets/judge_eval/judge_test.json') as f:
    test_data = json.load(f)

print(f'final: {len(final_data)} examples')
print(f'train+val+test: {len(train_data) + len(val_data) + len(test_data)} examples')

# Get unique inputs from each
final_inputs = set(item['input'] for item in final_data)
split_inputs = set()

for item in train_data:
    split_inputs.add(item['messages'][0]['content'])
for item in val_data:
    split_inputs.add(item['messages'][0]['content'])
for item in test_data:
    split_inputs.add(item['messages'][0]['content'])

print(f'\nUnique inputs:')
print(f'  final: {len(final_inputs)}')
print(f'  splits: {len(split_inputs)}')

# Check overlap
only_in_final = final_inputs - split_inputs
only_in_splits = split_inputs - final_inputs

if only_in_final:
    print(f'\n⚠️  {len(only_in_final)} examples ONLY in final (not in splits)')
    print('\nFirst few examples only in final:')
    for i, inp in enumerate(list(only_in_final)[:3]):
        print(f'{i+1}. {inp[:100]}...')
        
if only_in_splits:
    print(f'\n⚠️  {len(only_in_splits)} examples ONLY in splits (not in final)')
    print('\nFirst few examples only in splits:')
    for i, inp in enumerate(list(only_in_splits)[:3]):
        print(f'{i+1}. {inp[:100]}...')
    
if not only_in_final and not only_in_splits:
    print('\n✅ Same examples! Just different split ratios.')
else:
    print('\n❌ Different data! Need to merge before splitting.')
