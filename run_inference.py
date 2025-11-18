#!/usr/bin/env python3
"""
Inference script for running translations with trained LoRA adapters.

Usage examples:
    # Interactive mode (best for testing multiple prompts)
    python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --interactive
    
    # Single prompt
    python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --prompt "Hello, how are you?"
    
    # Batch mode from file
    python run_inference.py --adapter ./adapters/20251118_124144_best_ep1 --input prompts.txt --output translations.txt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pt_app.trainer.inference import TranslationInference


def run_single_prompt(translator, prompt, args):
    """Run inference on a single prompt"""
    result = translator.translate(
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        use_quality_filter=args.quality_filter,
        verbose=args.verbose
    )
    
    print(f"\nEnglish:    {prompt}")
    print(f"Portuguese: {result['filtered']}")
    
    if args.verbose and result['raw'] != result['filtered']:
        print(f"\nRaw output: {result['raw']}")
        print("[Quality filter was applied]")


def run_batch_inference(translator, input_file, output_file, args):
    """Run inference on multiple prompts from a file"""
    print(f"\n[INFO] Reading prompts from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"[INFO] Processing {len(prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Translating...", end='\r', flush=True)
        result = translator.translate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_quality_filter=args.quality_filter,
            verbose=False
        )
        results.append({
            'source': prompt,
            'translation': result['filtered']
        })
    
    print("\n[INFO] Writing translations to:", output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"EN: {item['source']}\n")
            f.write(f"PT: {item['translation']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"[SUCCESS] Completed {len(results)} translations")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--adapter', 
        type=str, 
        required=True,
        help='Path to adapter directory (e.g., ./adapters/20251118_124144_best_ep1)'
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive inference mode'
    )
    mode_group.add_argument(
        '--prompt',
        type=str,
        help='Single prompt to translate'
    )
    mode_group.add_argument(
        '--input',
        type=str,
        help='Input file with prompts (one per line)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch translations (required with --input)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Maximum tokens to generate (default: 150)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--no-filter',
        dest='quality_filter',
        action='store_false',
        help='Disable quality filtering (not recommended)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU inference (overrides auto-detection)'
    )
    parser.add_argument(
        '--no-cpu',
        dest='force_no_cpu',
        action='store_true',
        help='Force use of GPU/MPS even if auto-detection suggests CPU'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed filtering information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if not Path(args.adapter).exists():
        parser.error(f"Adapter path does not exist: {args.adapter}")
    
    # Initialize translator
    print("\n" + "="*80)
    print("INITIALIZING INFERENCE")
    print("="*80)
    translator = TranslationInference()
    
    # Load adapter with appropriate CPU setting
    if args.cpu:
        use_cpu = True
    elif args.force_no_cpu:
        use_cpu = False
    else:
        use_cpu = None  # Let auto-detection decide
    
    translator.load_adapter(
        adapter_path=args.adapter,
        use_cpu=use_cpu
    )
    
    # Run inference based on mode
    if args.interactive:
        translator.interactive(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_quality_filter=args.quality_filter
        )
    elif args.prompt:
        run_single_prompt(translator, args.prompt, args)
    elif args.input:
        run_batch_inference(translator, args.input, args.output, args)


if __name__ == "__main__":
    main()
