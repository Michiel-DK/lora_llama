#!/usr/bin/env python3
"""Test locally downloaded adapter on a few examples"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def test_adapter(
    adapter_path="./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20251209_150013_final",
    base_model="Qwen/Qwen2.5-3B-Instruct",
):
    print("="*80)
    print("TESTING LOCAL ADAPTER")
    print("="*80)
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")
    
    # Force CPU (MPS has issues with this model)
    device = torch.device("cpu")
    print("Device: CPU (forced - MPS not compatible)")
    
    # Load base model (CPU only)
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,  # CPU requires float32
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    
    # Load adapter
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n✅ Model loaded successfully\n")
    
    # Test examples
    test_cases = [
        {
            "source": "He graduated from the College of Arts & Sciences of the University of Virginia in 1950.",
            "translation": "Ele se formou no College of Arts & Sciences da Universidade de Virgínia em 1950.",
            "reference": "Ele se formou pelo College of Arts & Sciences da Universidade de Virgínia em 1950.",
        },
        {
            "source": "The cat is sleeping on the couch.",
            "translation": "O gato está dormindo no sofá.",
            "reference": "O gato está dormindo no sofá.",
        },
        {
            "source": "I love programming in Python.",
            "translation": "Eu amo programando em Python.",
            "reference": "Eu amo programar em Python.",
        },
    ]
    
    print("="*80)
    print("TESTING ON EXAMPLES")
    print("="*80)
    
    for i, example in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        print(f"Source: {example['source']}")
        print(f"Translation: {example['translation']}")
        print(f"Reference: {example['reference']}")
        print(f"\n{'─'*80}")
        
        # Create prompt
        prompt = f"""Evaluate this English to Portuguese translation.

Source: {example['source']}
Translation: {example['translation']}
Reference: {example['reference']}

Provide a score (0-10) and detailed feedback."""
        
        input_text = f"User: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        print("Generating prediction...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = generated_text.split("Assistant:")[-1].strip()
        
        print(f"\n🤖 MODEL PREDICTION:")
        print(prediction)
    
    print(f"\n{'='*80}")
    print("✅ Testing complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test locally downloaded adapter")
    parser.add_argument(
        "--adapter_path",
        default="./adapters_eval/Qwen2.5-3B-Instruct-judge-3ep-20251209_150013_final",
        help="Path to downloaded adapter"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name"
    )
    
    args = parser.parse_args()
    test_adapter(args.adapter_path, args.base_model)
