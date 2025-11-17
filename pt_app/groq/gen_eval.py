import os
import json
import time
import argparse
from datetime import datetime, timedelta
from datasets import load_dataset
from groq import Groq
from tqdm import tqdm

# Rate limiting configuration for FREE TIER
# Using llama-3.3-70b-versatile: 30 RPM, 14.4K RPD, 12K TPM
REQUESTS_PER_MINUTE = 25  # Slightly under limit for safety
REQUESTS_PER_DAY = 14000  # Slightly under limit
TOKENS_PER_MINUTE = 10000  # Conservative estimate
MIN_DELAY_BETWEEN_REQUESTS = 2.5  # seconds (60/25 = 2.4)

class RateLimiter:
    """Track and enforce rate limits"""
    def __init__(self):
        self.request_times = []
        self.minute_request_count = 0
        self.minute_reset_time = datetime.now() + timedelta(minutes=1)
        self.total_requests = 0
        self.rate_limit_headers = []
    
    def wait_for_rate_limit(self):
        """Smart rate limiting that respects both RPM and RPD"""
        current_time = datetime.now()
        
        # Reset minute counter if needed
        if current_time >= self.minute_reset_time:
            self.minute_request_count = 0
            self.minute_reset_time = current_time + timedelta(minutes=1)
        
        # Check if we need to wait for RPM
        if self.minute_request_count >= REQUESTS_PER_MINUTE:
            wait_seconds = (self.minute_reset_time - current_time).total_seconds()
            if wait_seconds > 0:
                print(f"  [Rate limit] Waiting {wait_seconds:.1f}s for RPM reset...")
                time.sleep(wait_seconds + 0.5)
                self.minute_request_count = 0
                self.minute_reset_time = datetime.now() + timedelta(minutes=1)
        
        # Enforce minimum delay between requests
        if self.request_times:
            time_since_last = (current_time - self.request_times[-1]).total_seconds()
            if time_since_last < MIN_DELAY_BETWEEN_REQUESTS:
                sleep_time = MIN_DELAY_BETWEEN_REQUESTS - time_since_last
                time.sleep(sleep_time)
        
        # Track this request
        self.request_times.append(datetime.now())
        self.minute_request_count += 1
        self.total_requests += 1
        
        # Keep only last 60 seconds of requests for tracking
        cutoff = datetime.now() - timedelta(seconds=60)
        self.request_times = [t for t in self.request_times if t > cutoff]
    
    def add_headers(self, response_headers):
        """Store rate limit headers from response"""
        headers = {
            'remaining_requests': response_headers.get('x-ratelimit-remaining-requests'),
            'remaining_tokens': response_headers.get('x-ratelimit-remaining-tokens'),
            'limit_requests': response_headers.get('x-ratelimit-limit-requests'),
            'limit_tokens': response_headers.get('x-ratelimit-limit-tokens'),
            'reset_requests': response_headers.get('x-ratelimit-reset-requests'),
            'reset_tokens': response_headers.get('x-ratelimit-reset-tokens'),
        }
        self.rate_limit_headers.append(headers)
    
    def get_latest_limits(self):
        """Get most recent rate limit info"""
        if self.rate_limit_headers:
            return self.rate_limit_headers[-1]
        return None

def generate_evaluation_examples(client, rate_limiter, source_en, reference_pt, max_retries=3):
    """
    Generate 4 translation variations with different quality levels
    Returns list of evaluation examples
    """
    
    prompt = f"""You are a translation quality assessment expert for English to Portuguese translation.

                Given this professional translation:
                Source (English): {source_en}
                Reference (Portuguese): {reference_pt}

                Generate exactly 4 Portuguese translations with different quality levels:

                1. POOR QUALITY (Score 3-5/10):
                - Major grammar error (wrong verb tense, wrong subject-verb agreement)
                - OR missing critical word
                - OR completely wrong word choice
                
                2. MEDIOCRE QUALITY (Score 5-7/10):
                - Minor error: wrong preposition, missing article, or extra word
                - Meaning preserved but not natural
                - Awkward phrasing
                
                3. GOOD QUALITY (Score 7-9/10):
                - Mostly correct but slightly unnatural
                - Minor word order issue OR overly literal translation
                - Still understandable and acceptable
                
                4. EXCELLENT QUALITY (Score 9-10/10):
                - Alternative translation that's equally good or better
                - Natural and fluent Portuguese
                - Different word choice but perfectly valid

                For EACH translation, provide:
                - translation: The Portuguese text
                - score: Number from 0-10
                - issues: List of specific problems (empty list if none)
                - feedback: 1-2 sentences explaining the assessment

                Return ONLY a valid JSON array with no additional text:
                [
                {{
                    "translation": "Portuguese text here",
                    "score": 5,
                    "issues": ["wrong verb", "missing preposition"],
                    "feedback": "Uses 'gosto' (like) instead of 'adoro' (love). Missing preposition 'de'."
                }},
                ...
                ]"""

    for attempt in range(max_retries):
        try:
            # Wait for rate limit before making request
            rate_limiter.wait_for_rate_limit()
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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
            
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"  No JSON array found (attempt {attempt + 1})")
                continue
                
            json_str = content[start_idx:end_idx]
            variations = json.loads(json_str)
            
            # Validate we got 4 variations
            if len(variations) != 4:
                print(f"  Got {len(variations)} variations instead of 4 (attempt {attempt + 1})")
                continue
            
            # Validate structure
            required_keys = {"translation", "score", "issues", "feedback"}
            if all(required_keys.issubset(v.keys()) for v in variations):
                return variations
            else:
                print(f"  Missing required keys (attempt {attempt + 1})")
                continue
                
        except json.JSONDecodeError as e:
            print(f"  JSON decode error (attempt {attempt + 1}): {str(e)[:50]}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
        except Exception as e:
            # Check if it's a rate limit error
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"  Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                continue
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

def main(flores_file, num_samples=None, output_file="judge_training_data.json"):
    """
    Main function to generate evaluation data
    
    Args:
        flores_file: Path to FLORES dataset JSON file
        num_samples: Number of FLORES examples to process (None = all)
        output_file: Output filename for the training data
    """
    # Initialize
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    rate_limiter = RateLimiter()
    
    # Load FLORES dataset from file
    print(f"Loading FLORES dataset from: {flores_file}")
    with open(flores_file, "r", encoding="utf-8") as f:
        flores_data = json.load(f)
    
    print(f"✓ Loaded {len(flores_data)} FLORES examples")
    
    # Determine sample size
    total_available = len(flores_data)
    if num_samples is None:
        sample_size = total_available
    else:
        sample_size = min(num_samples, total_available)
    
    print(f"Will process: {sample_size} examples")
    
    # Estimate time based on rate limits
    estimated_minutes = (sample_size / REQUESTS_PER_MINUTE) + 5
    estimated_hours = estimated_minutes / 60
    
    print("\n" + "="*60)
    print("GENERATION PLAN")
    print("="*60)
    print(f"FLORES examples to process: {sample_size}")
    print(f"Expected evaluation examples: ~{sample_size * 4}")
    print(f"Rate limit: {REQUESTS_PER_MINUTE} requests/minute")
    print(f"Estimated time: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"Daily limit usage: ~{sample_size}/{REQUESTS_PER_DAY} ({sample_size/REQUESTS_PER_DAY*100:.1f}%)")
    print("="*60)
    print("\nStarting generation...")
    print("(Progress saves every 100 examples)")
    print("="*60 + "\n")
    
    # Generate evaluation dataset
    evaluation_data = []
    failed_count = 0
    start_time = datetime.now()
    
    for idx in tqdm(range(sample_size), desc="Generating"):
        example = flores_data[idx]
        
        # Extract source and reference from example
        source_en = example['sentence_eng_Latn']
        reference_pt = example['sentence_por_Latn']
        
        # Generate variations
        variations = generate_evaluation_examples(client, rate_limiter, source_en, reference_pt)
        
        if variations is None:
            failed_count += 1
            continue
        
        # Create evaluation examples for each variation
        for variation in variations:
            eval_example = format_evaluation_example(source_en, reference_pt, variation)
            evaluation_data.append(eval_example)
        
        # Progress report every 50 examples
        if (idx + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (sample_size - idx - 1) / rate if rate > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"Progress: {idx + 1}/{sample_size} ({(idx+1)/sample_size*100:.1f}%)")
            print(f"Generated: {len(evaluation_data)} examples")
            print(f"Failed: {failed_count}")
            print(f"Rate: {rate:.1f} FLORES/min")
            print(f"Elapsed: {elapsed:.1f} min | Remaining: ~{remaining:.1f} min")
            print(f"{'='*60}")
        
        # Save progress every 100 examples
        if (idx + 1) % 100 == 0:
            progress_file = f"judge_data_progress_{idx + 1}.json"
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
            print(f"  → Saved progress to {progress_file}")
    
    # Final statistics
    elapsed_time = (datetime.now() - start_time).total_seconds()
    elapsed_minutes = elapsed_time / 60
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Total time: {elapsed_minutes:.1f} minutes ({elapsed_minutes/60:.1f} hours)")
    print(f"FLORES examples processed: {sample_size}")
    print(f"Evaluation examples generated: {len(evaluation_data)}")
    print(f"Failed generations: {failed_count}")
    print(f"Success rate: {(sample_size - failed_count) / sample_size * 100:.1f}%")
    print(f"Average rate: {sample_size / elapsed_minutes:.1f} FLORES/min")
    print("="*60)
    
    # Save final dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved evaluation data to: {output_file}")
    
    # Save statistics
    stats = {
        "generation_date": datetime.now().isoformat(),
        "total_flores_examples": sample_size,
        "total_evaluation_examples": len(evaluation_data),
        "failed_generations": failed_count,
        "success_rate": (sample_size - failed_count) / sample_size,
        "examples_per_flores": len(evaluation_data) / (sample_size - failed_count) if (sample_size - failed_count) > 0 else 0,
        "time_seconds": elapsed_time,
        "time_minutes": elapsed_minutes,
        "rate_per_minute": sample_size / elapsed_minutes,
        "model_used": "llama-3.3-70b-versatile",
        "rate_limit_rpm": REQUESTS_PER_MINUTE,
        "total_api_calls": rate_limiter.total_requests,
    }
    
    with open("generation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to: generation_stats.json")
    
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
            output_file=args.output
        )
        print("\n✓ Generation completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        print("Progress has been saved. You can resume or use partial data.")
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        raise