from transformers import TrainerCallback

class TranslationCallback(TrainerCallback):
    """Test translation quality during training"""
    
    def __init__(self, tokenizer, test_sentences=None):
        self.tokenizer = tokenizer
        self.test_sentences = test_sentences or [
            "Hello, how are you?",
            "The weather is nice today.",
        ]
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run after each evaluation"""
        if model is None:
            return
        
        print("\n[TRANSLATION TEST]")
        for sentence in self.test_sentences:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that translates English text to Portuguese.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            print(f"EN: {sentence}")
            print(f"PT: {response}")
            print("-" * 40)

# Add to trainer
callbacks = [TranslationCallback(tokenizer)]