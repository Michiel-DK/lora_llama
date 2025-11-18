"""
Example script showing how to use the TranslationInference class.

This demonstrates the main inference methods:
1. load_adapter() - Load a trained adapter
2. translate() - Translate a single prompt
3. batch_translate() - Translate multiple prompts
4. interactive() - Interactive translation session
"""

from pt_app.trainer.inference import TranslationInference


def example_single_inference():
    """Example: Single prompt translation"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Prompt Inference")
    print("="*80)
    
    # Initialize translator
    translator = TranslationInference()
    
    # Load your trained adapter
    adapter_path = "./adapters/20251118_124144_best_ep1"  # Update with your adapter path
    translator.load_adapter(adapter_path, use_cpu=True)
    
    # Translate a single prompt
    prompt = "Hello, how are you today?"
    result = translator.translate(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.8,
        use_quality_filter=True
    )
    
    print(f"\nEnglish:    {prompt}")
    print(f"Portuguese: {result['filtered']}")


def example_multiple_prompts():
    """Example: Multiple prompts in a loop"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Prompts")
    print("="*80)
    
    # Initialize translator and load adapter
    translator = TranslationInference()
    adapter_path = "./adapters/20251118_124144_best_ep1"  # Update with your adapter path
    translator.load_adapter(adapter_path, use_cpu=True)
    
    # List of prompts to translate
    prompts = [
        "Good morning!",
        "Where is the library?",
        "I would like a cup of coffee, please.",
        "The weather is beautiful today.",
    ]
    
    print("\nTranslating multiple prompts:\n")
    for i, prompt in enumerate(prompts, 1):
        result = translator.translate(prompt)
        print(f"{i}. EN: {prompt}")
        print(f"   PT: {result['filtered']}\n")


def example_batch_translate():
    """Example: Batch translation with progress bar"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Translation")
    print("="*80)
    
    # Initialize translator and load adapter
    translator = TranslationInference()
    adapter_path = "./adapters/20251118_124144_best_ep1"  # Update with your adapter path
    translator.load_adapter(adapter_path, use_cpu=True)
    
    # List of prompts
    prompts = [
        "Good morning!",
        "Where is the library?",
        "I would like a cup of coffee, please.",
        "The weather is beautiful today.",
        "Thank you very much!",
    ]
    
    # Batch translate with progress bar
    results = translator.batch_translate(
        prompts=prompts,
        temperature=0.8,
        use_quality_filter=True,
        show_progress=True
    )
    
    print("\nResults:")
    for prompt, result in zip(prompts, results):
        print(f"EN: {prompt}")
        print(f"PT: {result['filtered']}\n")


def example_interactive():
    """Example: Interactive mode"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Interactive Mode")
    print("="*80)
    
    # Initialize translator and load adapter
    translator = TranslationInference()
    adapter_path = "./adapters/20251118_124144_best_ep1"  # Update with your adapter path
    translator.load_adapter(adapter_path, use_cpu=True)
    
    # Start interactive session
    translator.interactive(
        max_new_tokens=150,
        temperature=0.8,
        use_quality_filter=True
    )


def example_custom_parameters():
    """Example: Using custom generation parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Parameters")
    print("="*80)
    
    translator = TranslationInference()
    adapter_path = "./adapters/20251118_124144_best_ep1"  # Update with your adapter path
    translator.load_adapter(adapter_path, use_cpu=True)
    
    prompt = "This is a test sentence."
    
    # More deterministic (lower temperature)
    print("\n--- Lower temperature (0.3) - more deterministic ---")
    result1 = translator.translate(
        prompt=prompt,
        temperature=0.3,
        use_quality_filter=True
    )
    print(f"PT: {result1['filtered']}")
    
    # More creative (higher temperature)
    print("\n--- Higher temperature (1.2) - more creative ---")
    result2 = translator.translate(
        prompt=prompt,
        temperature=1.2,
        use_quality_filter=True
    )
    print(f"PT: {result2['filtered']}")
    
    # Without quality filter
    print("\n--- Without quality filter ---")
    result3 = translator.translate(
        prompt=prompt,
        temperature=0.8,
        use_quality_filter=False
    )
    print(f"PT: {result3['raw']}")


if __name__ == "__main__":
    import sys
    
    print("\nAvailable examples:")
    print("1. Single prompt inference")
    print("2. Multiple prompts (loop)")
    print("3. Batch translation")
    print("4. Interactive mode")
    print("5. Custom parameters")
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = input("\nEnter example number (1-5) or press Enter for example 1: ").strip()
        if not example_num:
            example_num = "1"
    
    examples = {
        "1": example_single_inference,
        "2": example_multiple_prompts,
        "3": example_batch_translate,
        "4": example_interactive,
        "5": example_custom_parameters,
    }
    
    if example_num in examples:
        examples[example_num]()
    else:
        print("Invalid example number. Please choose 1-5.")
