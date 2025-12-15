#!/usr/bin/env python3
"""
Analyze test predictions from eval results.
Filter and explore predictions by score difference and extraction status.
"""
import json
import pandas as pd
import argparse


def load_predictions(json_file):
    """Load predictions from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    return predictions


def analyze_predictions(predictions):
    """
    Analyze predictions and create a DataFrame with useful columns.
    
    Returns:
        DataFrame with predictions and computed metrics
    """
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    # Add score difference column
    df['score_diff'] = abs(df['predicted_score'] - df['reference_score'])
    
    # Add absolute difference category
    df['diff_category'] = pd.cut(
        df['score_diff'],
        bins=[0, 1, 2, 3, 5, 10],
        labels=['Excellent (≤1)', 'Good (1-2)', 'Fair (2-3)', 'Poor (3-5)', 'Very Poor (>5)']
    )
    
    # Add extraction status
    df['extraction_ok'] = df['score_extracted'].fillna(False)
    
    # Add predicted/reference for failed extractions
    df['predicted_score'] = df['predicted_score'].fillna(-1)
    df['reference_score'] = df['reference_score'].fillna(-1)
    
    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal predictions: {len(df)}")
    print(f"Successful extractions: {df['extraction_ok'].sum()} ({df['extraction_ok'].mean()*100:.1f}%)")
    print(f"Failed extractions: {(~df['extraction_ok']).sum()} ({(~df['extraction_ok']).mean()*100:.1f}%)")
    
    # Stats on successful predictions only
    successful = df[df['extraction_ok']]
    if len(successful) > 0:
        print(f"\n--- Successful Predictions Statistics ---")
        print(f"Mean absolute error: {successful['score_diff'].mean():.2f}")
        print(f"Median absolute error: {successful['score_diff'].median():.2f}")
        print(f"Max error: {successful['score_diff'].max():.2f}")
        print(f"Min error: {successful['score_diff'].min():.2f}")
        
        print(f"\n--- Error Distribution ---")
        print(successful['diff_category'].value_counts().sort_index())
    
    print("\n" + "="*80)


def filter_predictions(df, 
                       min_diff=None, 
                       max_diff=None,
                       extraction_failed_only=False,
                       extraction_success_only=False):
    """
    Filter predictions based on criteria.
    
    Args:
        df: DataFrame with predictions
        min_diff: Minimum score difference to include
        max_diff: Maximum score difference to include
        extraction_failed_only: Show only failed extractions
        extraction_success_only: Show only successful extractions
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if extraction_failed_only:
        filtered = filtered[~filtered['extraction_ok']]
    elif extraction_success_only:
        filtered = filtered[filtered['extraction_ok']]
    
    if min_diff is not None:
        filtered = filtered[filtered['score_diff'] >= min_diff]
    
    if max_diff is not None:
        filtered = filtered[filtered['score_diff'] <= max_diff]
    
    return filtered


def display_predictions(df, num_examples=10):
    """Display predictions in readable format."""
    print(f"\nShowing {min(len(df), num_examples)} of {len(df)} predictions:\n")
    
    for idx, row in df.head(num_examples).iterrows():
        print("="*80)
        print(f"Example {idx + 1}")
        print("="*80)
        
        print(f"\n📝 USER PROMPT:")
        print(row['user_prompt'][:200] + "..." if len(row['user_prompt']) > 200 else row['user_prompt'])
        
        print(f"\n✅ REFERENCE (Score: {row['reference_score']:.1f}):")
        print(row['reference_response'][:150] + "..." if len(row['reference_response']) > 150 else row['reference_response'])
        
        print(f"\n🤖 PREDICTED (Score: {row['predicted_score']:.1f}):")
        print(row['predicted_response'][:150] + "..." if len(row['predicted_response']) > 150 else row['predicted_response'])
        
        if row['extraction_ok']:
            print(f"\n📊 Score Difference: {row['score_diff']:.2f} ({row['diff_category']})")
        else:
            print(f"\n⚠️  EXTRACTION FAILED")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze test predictions")
    parser.add_argument(
        '--file',
        default='./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20251209_150013_final/test_predictions_fast.json',
        help='Path to predictions JSON file'
    )
    parser.add_argument('--min-diff', type=float, help='Minimum score difference')
    parser.add_argument('--max-diff', type=float, help='Maximum score difference')
    parser.add_argument('--failed-only', action='store_true', help='Show only failed extractions')
    parser.add_argument('--success-only', action='store_true', help='Show only successful extractions')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to display')
    parser.add_argument('--save-csv', type=str, help='Save filtered results to CSV')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.file}")
    predictions = load_predictions(args.file)
    
    # Analyze
    df = analyze_predictions(predictions)
    
    # Print summary
    print_summary(df)
    
    # Filter
    filtered = filter_predictions(
        df,
        min_diff=args.min_diff,
        max_diff=args.max_diff,
        extraction_failed_only=args.failed_only,
        extraction_success_only=args.success_only
    )
    
    print(f"\n🔍 Filtered to {len(filtered)} predictions")
    
    # Display
    display_predictions(filtered, num_examples=args.num_examples)
    
    # Save if requested
    if args.save_csv:
        filtered.to_csv(args.save_csv, index=False)
        print(f"\n✅ Saved filtered results to: {args.save_csv}")


if __name__ == "__main__":
    main()
