import os
import json
import time
import argparse
import random
from datetime import datetime, timedelta
from datasets import load_dataset
from groq import Groq
from tqdm import tqdm

# Import prompt templates from separate file
from prompts import PROMPT_BALANCED, PROMPT_LOW_SCORES, PROMPT_MIDDLE_RANGE

# Rate limiting configuration for FREE TIER
REQUESTS_PER_MINUTE = 50  # Reduced from 25 for safety
REQUESTS_PER_DAY = 14000
TOKENS_PER_MINUTE = 10000
MIN_DELAY_BETWEEN_REQUESTS = 1.5  # Increased from 2.5 for safety

class RateLimiter:
    """Track and enforce rate limits"""
    def __init__(self):
        self.request_times = []
        self.total_requests = 0
        self.rate_limit_headers = []
    
    def wait_for_rate_limit(self):
        """Enforce rate limit with proper tracking"""
        current_time = time.time()
        
        # Remove requests older than 60 seconds
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # DEBUG: Print current state
        print(f"  [DEBUG] Requests in last 60s: {len(self.request_times)}")
        
        # Check if we're at the limit
        if len(self.request_times) >= REQUESTS_PER_MINUTE:
            # Calculate how long to wait
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request) + 2  # +2 for safety buffer
            
            if wait_time > 0:
                print(f"  [Rate limit] Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Clear old requests after waiting
                self.request_times = []
        
        # Enforce minimum delay between requests
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < MIN_DELAY_BETWEEN_REQUESTS:
                sleep_time = MIN_DELAY_BETWEEN_REQUESTS - time_since_last
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(time.time())
        self.total_requests += 1
    
    def add_headers(self, headers):
        """Store rate limit headers from API response"""
        self.rate_limit_headers.append(headers)
    
    def get_latest_limits(self):
        """Get latest rate limit information from headers"""
        if not self.rate_limit_headers:
            return None
        return self.rate_limit_headers[-1]

def generate_evaluation_examples(client, rate_limiter, source_en, reference_pt, prompt_type="balanced", max_retries=3):
    """
    Generate 4 translation variations with different quality levels
    
    Args:
        client: Groq client
        rate_limiter: RateLimiter instance
        source_en: English source text
        reference_pt: Portuguese reference translation
        prompt_type: Either 'balanced', 'low_scores', or 'middle_range'
        max_retries: Number of retry attempts
    
    Returns:
        List of evaluation examples
    """
    
    # Select prompt template based on type
    if prompt_type == "low_scores":
        prompt_template = PROMPT_LOW_SCORES
    elif prompt_type == "middle_range":
        prompt_template = PROMPT_MIDDLE_RANGE
    else:
        prompt_template = PROMPT_BALANCED
    
    prompt = prompt_template.format(source_en=source_en, reference_pt=reference_pt)
    
    # Retry loop for API call 

    for attempt in range(max_retries):
        try:
            # CRITICAL: Wait for rate limit BEFORE making request
            rate_limiter.wait_for_rate_limit()
            
            response = client.chat.completions.create(
                #model="llama-3.3-70b-versatile",
                model = 'qwen/qwen3-32b',
                messages=[
                    {"role": "system", "content": "You are a translation quality expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            # Store rate limit headers
            if hasattr(response, 'headers'):
                rate_limiter.add_headers(response.headers)
            
            content = response.choices[0].message.content.strip()
            
            # Remove thinking tokens if present
            if '<think>' in content:
                # Remove everything from <think> to </think>
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
            
            # Remove markdown code fences if present
            if content.startswith('```'):
                # Remove ```json or ``` at start and ``` at end
                import re
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
            
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"  No JSON array found (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
            json_str = content[start_idx:end_idx]
            variations = json.loads(json_str)
            
            # Validate we got 4 variations
            if len(variations) != 4:
                print(f"  Got {len(variations)} variations instead of 4 (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
            
            # Validate structure
            required_keys = {"translation", "score", "issues", "feedback"}
            if all(required_keys.issubset(v.keys()) for v in variations):
                return variations
            else:
                print(f"  Missing required keys (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
                
        except json.JSONDecodeError as e:
            print(f"  JSON decode error (attempt {attempt + 1}): {e}")
            print(f"  Response preview: {content[:500] if 'content' in locals() else 'No content'}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a rate limit error
            if "429" in str(e) or "rate_limit" in error_msg or "too many requests" in error_msg:
                print(f"  API rate limit hit! Waiting 65 seconds...")
                time.sleep(65)  # Wait longer than 60s
                # Don't count this as an attempt
                continue
            else:
                print(f"  Error (attempt {attempt + 1}): {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
    
    return None

def format_evaluation_example(source, reference, variation):
    """Format a single evaluation example for judge training"""
    return {
        "input": f"""Evaluate this English to Portuguese translation.

Source: {source}
Translation: {variation['translation']}
Reference: {reference}

Provide a score (0-10) and detailed feedback.""",
        "output": f"""Score: {variation['score']}/10

Issues: {', '.join(variation['issues']) if variation['issues'] else 'None'}

Feedback: {variation['feedback']}

{f"The reference translation is: {reference}" if variation['score'] < 8 else "This is a high-quality translation."}"""
    }

def print_rate_limit_summary(rate_limiter, elapsed_time):
    """Print comprehensive rate limit usage summary"""
    print("\n" + "="*60)
    print("RATE LIMIT SUMMARY")
    print("="*60)
    
    # Latest headers
    latest = rate_limiter.get_latest_limits()
    if latest:
        print("\nCurrent Status:")
        print(f"  Remaining requests (RPD): {latest['remaining_requests']} / {latest['limit_requests']}")
        print(f"  Remaining tokens (TPM): {latest['remaining_tokens']} / {latest['limit_tokens']}")
        print(f"  Requests reset in: {latest['reset_requests']}")
        print(f"  Tokens reset in: {latest['reset_tokens']}")
        
        # Calculate usage
        if latest['limit_requests']:
            used_requests = int(latest['limit_requests']) - int(latest['remaining_requests'])
            usage_pct = (used_requests / int(latest['limit_requests'])) * 100
            print(f"\n  Daily usage: {used_requests}/{latest['limit_requests']} ({usage_pct:.1f}%)")
    
    # Session statistics
    print(f"\nSession Statistics:")
    print(f"  Total API calls made: {rate_limiter.total_requests}")
    print(f"  Average rate: {rate_limiter.total_requests / (elapsed_time / 60):.1f} requests/min")
    print(f"  Configured limit: {REQUESTS_PER_MINUTE} requests/min")
    
    # Remaining capacity
    if latest and latest['remaining_requests']:
        remaining = int(latest['remaining_requests'])
        can_process = remaining // 1  # Assuming 1 request per FLORES example
        time_for_remaining = can_process / REQUESTS_PER_MINUTE
        print(f"\nRemaining Capacity Today:")
        print(f"  Can process ~{can_process} more FLORES examples")
        print(f"  Would take ~{time_for_remaining:.0f} more minutes")
    
    print("="*60)

def main(flores_file, num_samples=None, output_file="judge_training_data.json", start_from=0, prompt_type=PROMPT_BALANCED, shuffle=False, seed=None):
    """
    Main function to generate evaluation data
    
    Args:
        flores_file: Path to FLORES dataset JSON file
        num_samples: Number of FLORES examples to process (None = all)
        output_file: Output filename for the training data
        start_from: Starting index in FLORES dataset (for resuming)
        prompt_type: PROMPT_BALANCED (default) or PROMPT_LOW_SCORES
        shuffle: Whether to shuffle the dataset before processing
        seed: Random seed for shuffling (for reproducibility)
    """
    # Initialize
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    rate_limiter = RateLimiter()
    
    # Load FLORES dataset from file
    print(f"Loading FLORES dataset from: {flores_file}")
    with open(flores_file, "r", encoding="utf-8") as f:
        flores_data = json.load(f)
    
    print(f"✓ Loaded {len(flores_data)} FLORES examples")
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            random.seed(seed)
            print(f"🔀 Shuffling dataset with seed: {seed}")
        else:
            print(f"🔀 Shuffling dataset randomly")
        random.shuffle(flores_data)
        print(f"✓ Dataset shuffled for diversity")
    
    # Determine sample size
    total_available = len(flores_data)
    
    # NEW: Handle start_from
    if start_from > 0:
        print(f"▶ Resuming from FLORES example {start_from}")
        if start_from >= total_available:
            print(f"ERROR: start_from ({start_from}) >= total examples ({total_available})")
            return None, None
    
    # Calculate end index
    if num_samples is None:
        end_idx = total_available
    else:
        end_idx = min(start_from + num_samples, total_available)
    
    sample_size = end_idx - start_from
    
    print(f"Will process: examples {start_from} to {end_idx-1} ({sample_size} total)")
    
    # Estimate time based on rate limits
    estimated_minutes = (sample_size / REQUESTS_PER_MINUTE) + 5
    estimated_hours = estimated_minutes / 60
    
    print("\n" + "="*60)
    print("GENERATION PLAN")
    print("="*60)
    print(f"FLORES range: {start_from} to {end_idx-1}")
    print(f"FLORES examples to process: {sample_size}")
    print(f"Expected evaluation examples: ~{sample_size * 4}")
    print(f"Rate limit: {REQUESTS_PER_MINUTE} requests/minute")
    print(f"Estimated time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"Daily limit usage: ~{sample_size}/{REQUESTS_PER_DAY} ({sample_size/REQUESTS_PER_DAY*100:.1f}%)")
    print("="*60)
    print("\nStarting generation...")
    print("(Progress saves every 50 examples)")
    print("="*60 + "\n")
    
    # Generate evaluation dataset
    evaluation_data = []
    failed_count = 0
    start_time = datetime.now()
    
    # CHANGED: Loop from start_from to end_idx
    for idx in tqdm(range(start_from, end_idx), desc="Generating"):
        example = flores_data[idx]
        
        # Extract source and reference from example
        source_en = example['sentence_eng_Latn']
        reference_pt = example['sentence_por_Latn']
        
        # Generate variations
        variations = generate_evaluation_examples(client, rate_limiter, source_en, reference_pt, prompt_type=prompt_type)
        
        if variations is None:
            failed_count += 1
            continue
        
        # Create evaluation examples for each variation
        for variation in variations:
            eval_example = format_evaluation_example(source_en, reference_pt, variation)
            evaluation_data.append(eval_example)
        
        # Save progress every 10 examples
        if (idx + 1) % 10 == 0:
            progress_file = f"judge_data_progress_{idx + 1}.json"
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        # Progress report every 50 examples
        if (idx + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            # CHANGED: Calculate based on actual progress from start_from
            processed = (idx + 1) - start_from
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (end_idx - idx - 1) / rate if rate > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"Progress: {idx + 1}/{end_idx} ({(idx + 1 - start_from)/sample_size*100:.1f}%)")
            print(f"Generated: {len(evaluation_data)} examples")
            print(f"Failed: {failed_count}")
            print(f"Rate: {rate:.1f} FLORES/min")
            print(f"Elapsed: {elapsed:.1f} min | Remaining: ~{remaining:.1f} min")
            print(f"Latest save: judge_data_progress_{(idx + 1) // 10 * 10}.json")
            print(f"{'='*60}")
        
    # Final statistics
    elapsed_time = (datetime.now() - start_time).total_seconds()
    elapsed_minutes = elapsed_time / 60
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Total time: {elapsed_minutes:.1f} minutes ({elapsed_minutes/60:.1f} hours)")
    print(f"FLORES range processed: {start_from} to {end_idx-1}")
    print(f"FLORES examples processed: {sample_size}")
    print(f"Evaluation examples generated: {len(evaluation_data)}")
    print(f"Failed generations: {failed_count}")
    print(f"Success rate: {(sample_size - failed_count) / sample_size * 100:.1f}%")
    print(f"Average rate: {sample_size / elapsed_minutes:.1f} FLORES/min")
    print("="*60)
    
    # CHANGED: Include range in output filename if starting from middle
    if start_from > 0 or end_idx < total_available:
        output_file = output_file.replace(".json", f"_{start_from}_{end_idx}.json")
    
    # Save final dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved evaluation data to: {output_file}")
    
    # Save statistics
    stats = {
        "generation_date": datetime.now().isoformat(),
        "flores_range": f"{start_from}-{end_idx-1}",
        "start_index": start_from,
        "end_index": end_idx,
        "total_flores_examples": sample_size,
        "total_evaluation_examples": len(evaluation_data),
        "failed_generations": failed_count,
        "success_rate": (sample_size - failed_count) / sample_size,
        "examples_per_flores": len(evaluation_data) / (sample_size - failed_count) if (sample_size - failed_count) > 0 else 0,
        "time_seconds": elapsed_time,
        "time_minutes": elapsed_minutes,
        "rate_per_minute": sample_size / elapsed_minutes,
        "model_used": "llama-3.1-8b-instant",
        "rate_limit_rpm": REQUESTS_PER_MINUTE,
        "total_api_calls": rate_limiter.total_requests,
    }
    
    stats_file = f"generation_stats_{start_from}_{end_idx}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to: {stats_file}")
    
    # Display sample examples
    print("\n" + "="*60)
    print("SAMPLE EVALUATION EXAMPLES")
    print("="*60)
    for i in range(min(2, len(evaluation_data))):
        print(f"\n--- Example {i + 1} ---")
        print(f"INPUT:\n{evaluation_data[i]['input']}\n")
        print(f"OUTPUT:\n{evaluation_data[i]['output']}")
        print("-"*60)
    
    # Print rate limit summary
    print_rate_limit_summary(rate_limiter, elapsed_time)
    
    return evaluation_data, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation training data for translation judge model using FLORES and Groq API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 50 examples (~2-3 minutes)
  python generate_eval_data.py --samples 50 --flores flores_en_pt_cache.json
  
  # Medium test with 200 examples (~8-10 minutes)
  python generate_eval_data.py --samples 200 --flores my_flores.json
  
  # Full dataset - all FLORES examples (~40 minutes)
  python generate_eval_data.py --flores flores_en_pt_cache.json
  
  # Custom output file
  python generate_eval_data.py --samples 100 --flores flores_en_pt_cache.json --output my_judge_data.json
        """
    )
    
    parser.add_argument(
        '--flores', '-f',
        type=str,
        required=True,
        help='Path to FLORES dataset JSON file (required)'
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=None,
        help='Number of FLORES examples to process (default: all examples in file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='judge_training_data.json',
        help='Output filename (default: judge_training_data.json)'
    )
    
    # NEW: Add start parameter
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this FLORES example index (default: 0, useful for resuming)"
    )
    
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["balanced", "low_scores", "middle_range"],
        default="balanced",
        help="Prompt type: 'balanced' (scores 3-10), 'low_scores' (scores 1-5), or 'middle_range' (scores 4-8)"
    )
    
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing for diversity (useful for small batches)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (for reproducibility)"
    )
    
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set")
        print("Please set it with: export GROQ_API_KEY='your_key_here'")
        exit(1)
    
    # Check if FLORES file exists
    if not os.path.exists(args.flores):
        print(f"ERROR: FLORES file not found: {args.flores}")
        exit(1)
    
    # Run generation
    try:
        evaluation_data, stats = main(
            flores_file=args.flores,
            num_samples=args.samples,
            output_file=args.output,
            start_from=args.start,
            prompt_type=args.prompt_type,
            shuffle=args.shuffle,
            seed=args.seed
        )
        print("\n✓ Generation completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        print("Progress has been saved. You can resume or use partial data.")
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        raise