import mlx.core as mx

def mlx_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    temp=0.1,
    top_p=0.9,
    repeat_penalty=1.1,
):
    """Generate text using MLX Llama model"""
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)
    
    # Make sure the model is in eval mode
    model.eval()
    
    # Start with prompt tokens
    prev_pos = 0
    generated_tokens = []
    
    # Generate tokens one by one
    for i in range(max_tokens):
        # Prepare input: original tokens + generated ones
        input_tokens = mx.array([tokens + generated_tokens])
        
        # Forward pass
        logits = model(input_tokens).squeeze()
        
        # Get the logits for the next token (last token in sequence)
        next_token_logits = logits[-1]
        
        # Apply temperature scaling
        if temp > 0:
            next_token_logits = next_token_logits / temp
        
        # Apply repetition penalty
        if repeat_penalty > 1.0 and len(generated_tokens) > 0:
            # For generated tokens, apply penalty by scaling down the logits
            for token in set(generated_tokens):
                next_token_logits[token] /= repeat_penalty
        
        # Apply top_p sampling
        if top_p < 1.0:
            # Sort in descending order - MLX doesn't have a descending parameter
            # so we need to sort and then reverse
            sorted_indices = mx.sort(mx.argsort(next_token_logits))[::-1]  # Sort and reverse
            sorted_logits = next_token_logits[sorted_indices]
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits), axis=0)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            
            # Create mask and apply it
            indices_to_remove = mx.zeros_like(next_token_logits, dtype=mx.bool)
            indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True
            next_token_logits = mx.where(
                indices_to_remove, -float("inf"), next_token_logits
            )
        
        # Sample from the logits
        probs = mx.softmax(next_token_logits)
        next_token = int(mx.random.categorical(probs, 1)[0])
        
        # Add to generated tokens
        generated_tokens.append(next_token)
        
        # Stop if we hit EOS token
        if next_token == tokenizer.eos_token_id:
            break
        
        # Update position (not needed in newer MLX implementations)
        prev_pos = prev_pos + input_tokens.shape[1]
    
    # Combine original and generated tokens
    all_tokens = tokens + generated_tokens
    
    # Decode to text
    return tokenizer.decode(all_tokens)


def mlx_generate_simple(
    model,
    tokenizer,
    prompt,
    max_tokens=100
):
    """Ultra-simplified greedy generation for MLX"""
    tokens = tokenizer.encode(prompt)
    model.eval()
    generated_tokens = []
    
    for i in range(max_tokens):
        # Create input array
        input_tokens = mx.array([tokens + generated_tokens])
        
        # Forward pass
        logits = model(input_tokens).squeeze()
        
        # Simple greedy decoding - just take highest probability token
        next_token = int(mx.argmax(logits[-1]).item())
        
        # Add to generated tokens
        generated_tokens.append(next_token)
        
        # Stop if we hit EOS token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Combine original and generated tokens
    all_tokens = tokens + generated_tokens
    
    # Decode to text
    return tokenizer.decode(all_tokens)

def mlx_generate_optimized(
    model,
    tokenizer,
    prompt,
    max_tokens=75  # Reduced from 100
):
    """Streamlined generation for MLX models"""
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    
    # Ensure model in eval mode
    model.eval()
    
    # Initialize
    generated_tokens = []
    
    # Simple autoregressive generation
    for _ in range(max_tokens):
        # Create input array with all tokens so far
        input_tokens = mx.array([tokens + generated_tokens])
        
        # Forward pass - get logits
        outputs = model(input_tokens)
        
        # Extract logits for next token prediction
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
            
        # Get the logits for the final position
        if len(logits.shape) > 2:
            next_token_logits = logits[0, -1, :]  # batch, sequence, vocab
        else:
            next_token_logits = logits[-1]  # just sequence, vocab
        
        # Simple greedy decoding
        next_token = int(mx.argmax(next_token_logits).item())
        
        # Add to generated tokens
        generated_tokens.append(next_token)
        
        # Stop if we hit EOS token or max length
        if next_token == tokenizer.eos_token_id:
            break
    
    # Combine and decode
    all_tokens = tokens + generated_tokens
    return tokenizer.decode(all_tokens)