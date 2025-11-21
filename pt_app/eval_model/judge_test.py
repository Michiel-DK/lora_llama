# test_judge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_judge(model_path="./qwen-3b-judge/final"):
    """Load trained judge model"""
    
    print(f"Loading judge from {model_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✓ Judge loaded")
    return model, tokenizer

def evaluate_translation(model, tokenizer, source, translation, reference=None):
    """
    Use judge to evaluate a translation
    """
    
    # Format input
    if reference:
        prompt = f"""Evaluate this English to Portuguese translation.

            Source: {source}
            Translation: {translation}
            Reference: {reference}

            Provide a score (0-10) and detailed feedback."""
    else:
        prompt = f"""Evaluate this English to Portuguese translation.

            Source: {source}
            Translation: {translation}

            Provide a score (0-10) and detailed feedback."""
                
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Generate evaluation
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response

if __name__ == "__main__":
    # Load judge
    model, tokenizer = load_judge()
    
    # Test cases
    test_cases = [
        {
            "source": "I love coffee",
            "translation": "Eu adoro café",
            "reference": "Eu adoro café"
        },
        {
            "source": "I love coffee",
            "translation": "Eu gosto café",  # Error: wrong verb, missing 'de'
            "reference": "Eu adoro café"
        },
        {
            "source": "The cat is sleeping",
            "translation": "O gato está dormindo",
            "reference": "O gato está dormindo"
        },
    ]
    
    print("\n" + "="*60)
    print("TESTING JUDGE")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Source: {test['source']}")
        print(f"Translation: {test['translation']}")
        print(f"Reference: {test['reference']}")
        print("\nJudge evaluation:")
        
        evaluation = evaluate_translation(
            model, tokenizer,
            test['source'],
            test['translation'],
            test['reference']
        )
        
        print(evaluation)
        print("-"*60)