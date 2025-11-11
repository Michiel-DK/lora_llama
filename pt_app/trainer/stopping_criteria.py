"""
Custom Stopping Criteria for Translation
Prevents repetitive generation and handles edge cases
"""

import torch
from transformers import StoppingCriteria
from typing import Optional
import re


class RepetitionStoppingCriteria(StoppingCriteria):
    """
    Stop generation when repetitive patterns are detected
    """
    
    def __init__(self, tokenizer, max_sentence_repeats: int = 1, 
                 check_ngram_repeats: bool = True,
                 check_language_switch: bool = True):
        """
        Args:
            tokenizer: Tokenizer for decoding
            max_sentence_repeats: Maximum times a sentence can repeat before stopping
            check_ngram_repeats: Whether to check for n-gram repetitions
            check_language_switch: Whether to stop on language switching
        """
        self.tokenizer = tokenizer
        self.max_sentence_repeats = max_sentence_repeats
        self.check_ngram_repeats = check_ngram_repeats
        self.check_language_switch = check_language_switch
        
        # Language detection keywords (simplified)
        self.english_keywords = {
            'the', 'is', 'are', 'was', 'were', 'will', 'have', 'has', 
            'this', 'that', 'what', 'when', 'where', 'how', 'why'
        }
        
        self.portuguese_keywords = {
            'o', 'a', 'os', 'as', 'é', 'são', 'foi', 'foram',
            'está', 'estão', 'tem', 'têm', 'que', 'de', 'em', 'para'
        }
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation should stop
        
        Args:
            input_ids: Generated token IDs (batch_size, sequence_length)
            scores: Model scores for next token prediction
            
        Returns:
            True if generation should stop, False otherwise
        """
        # Decode only the newly generated portion (after prompt)
        # We need to handle this carefully to not include the prompt
        try:
            # Decode the full sequence
            full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Check 1: Sentence repetition
            if self._has_sentence_repetition(full_text):
                return True
            
            # Check 2: N-gram repetition
            if self.check_ngram_repeats and self._has_ngram_repetition(full_text):
                return True
            
            # Check 3: Language switching
            if self.check_language_switch and self._has_language_switch(full_text):
                return True
            
            return False
            
        except Exception as e:
            # If something goes wrong, don't stop (let other criteria handle it)
            return False
    
    def _has_sentence_repetition(self, text: str) -> bool:
        """
        Check if the same sentence appears multiple times consecutively
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return False
        
        # Check last few sentences for repetition
        check_window = min(4, len(sentences))
        recent_sentences = sentences[-check_window:]
        
        # Normalize and count
        from collections import Counter
        normalized = [s.lower() for s in recent_sentences]
        counts = Counter(normalized)
        
        # If any sentence appears more than max_sentence_repeats
        for count in counts.values():
            if count > self.max_sentence_repeats:
                return True
        
        return False
    
    def _has_ngram_repetition(self, text: str) -> bool:
        """
        Check for repeated n-grams (3-5 words repeated consecutively)
        """
        words = text.split()
        
        if len(words) < 6:
            return False
        
        # Check last 15 words for repetitions
        recent_words = words[-15:] if len(words) > 15 else words
        
        # Check for 3-gram, 4-gram, and 5-gram repetitions
        for ngram_size in [5, 4, 3]:
            if len(recent_words) < ngram_size * 2:
                continue
            
            # Check if an n-gram repeats immediately
            for i in range(len(recent_words) - ngram_size * 2 + 1):
                ngram1 = ' '.join(recent_words[i:i+ngram_size])
                ngram2 = ' '.join(recent_words[i+ngram_size:i+ngram_size*2])
                
                if ngram1.lower() == ngram2.lower():
                    return True
        
        return False
    
    def _has_language_switch(self, text: str) -> bool:
        """
        Detect if generation switches from Portuguese to English mid-generation
        """
        # Only check last 20 words to detect recent switches
        words = text.split()
        if len(words) < 10:
            return False
        
        recent_words = words[-20:] if len(words) > 20 else words
        
        # Count language indicators
        en_count = sum(1 for w in recent_words if w.lower() in self.english_keywords)
        pt_count = sum(1 for w in recent_words if w.lower() in self.portuguese_keywords)
        
        # If we have significant English keywords in recent generation, it's likely a switch
        # (assuming target language is Portuguese)
        if en_count >= 3 and pt_count >= 2:
            # Both languages present - likely a switch
            return True
        
        return False


class MaxNewTokensStoppingCriteria(StoppingCriteria):
    """
    Stop after a maximum number of new tokens (not including prompt)
    """
    
    def __init__(self, prompt_length: int, max_new_tokens: int):
        """
        Args:
            prompt_length: Length of the input prompt in tokens
            max_new_tokens: Maximum new tokens to generate
        """
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if we've generated enough tokens
        """
        current_length = input_ids.shape[1]
        new_tokens = current_length - self.prompt_length
        
        return new_tokens >= self.max_new_tokens


class IncompleteOutputStoppingCriteria(StoppingCriteria):
    """
    Stop if output appears to be looping in an incomplete state
    """
    
    def __init__(self, tokenizer, check_after_tokens: int = 30):
        """
        Args:
            tokenizer: Tokenizer for decoding
            check_after_tokens: Start checking after this many new tokens
        """
        self.tokenizer = tokenizer
        self.check_after_tokens = check_after_tokens
        self.initial_length = None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation is stuck in incomplete patterns
        """
        if self.initial_length is None:
            self.initial_length = input_ids.shape[1]
            return False
        
        new_tokens = input_ids.shape[1] - self.initial_length
        
        if new_tokens < self.check_after_tokens:
            return False
        
        try:
            # Get last 10 tokens
            recent_tokens = input_ids[0][-10:]
            recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
            
            # IMPROVED PATTERNS - More specific!
            incomplete_patterns = [
                # Only prepositions followed by space (not part of word)
                r'\s(de|para|com|em|por|à|ao)\s*$',
                
                # Only standalone articles (with space before)
                r'\s(o|a|os|as|um|uma)\s*$',
                
                # Trailing comma
                r',\s*$',
            ]
            
            for pattern in incomplete_patterns:
                if re.search(pattern, recent_text):
                    # Check if this pattern has been present for the last several tokens
                    # (indicating stuck generation)
                    last_20_tokens = input_ids[0][-20:] if input_ids.shape[1] >= 20 else input_ids[0]
                    last_20_text = self.tokenizer.decode(last_20_tokens, skip_special_tokens=True)
                    
                    if re.search(pattern, last_20_text):
                        return True
            
            return False
            
        except Exception as e:
            return False


def create_stopping_criteria_list(tokenizer, prompt_length: Optional[int] = None,
                                  max_new_tokens: int = 100,
                                  prevent_repetition: bool = True,
                                  prevent_language_switch: bool = True,
                                  check_after_tokens: int = 100) :
    """
    Factory function to create a list of stopping criteria
    
    Args:
        tokenizer: Tokenizer for decoding
        prompt_length: Length of prompt in tokens (for max tokens criterion)
        max_new_tokens: Maximum new tokens to generate
        prevent_repetition: Whether to add repetition stopping
        prevent_language_switch: Whether to add language switch stopping
        
    Returns:
        StoppingCriteriaList with configured criteria
    """
    from transformers import StoppingCriteriaList
    
    criteria_list = StoppingCriteriaList()
    
    # Add repetition stopping
    if prevent_repetition:
        criteria_list.append(
            RepetitionStoppingCriteria(
                tokenizer=tokenizer,
                max_sentence_repeats=1,
                check_ngram_repeats=True,
                check_language_switch=prevent_language_switch
            )
        )
    
    # Add max tokens stopping
    if prompt_length is not None:
        criteria_list.append(
            MaxNewTokensStoppingCriteria(
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens
            )
        )
    
    # Add incomplete output detection
    criteria_list.append(
        IncompleteOutputStoppingCriteria(
            tokenizer=tokenizer,
            check_after_tokens=check_after_tokens
        )
    )
    
    return criteria_list


if __name__ == "__main__":
    # Test the stopping criteria
    from transformers import AutoTokenizer
    
    print("Testing RepetitionStoppingCriteria:\n")
    
    # Mock tokenizer for testing
    class MockTokenizer:
        def decode(self, tokens, skip_special_tokens=True):
            # Simulate decoded text
            return "Olá! Olá! Olá! Como está?"
    
    tokenizer = MockTokenizer()
    criteria = RepetitionStoppingCriteria(tokenizer)
    
    # Test with mock input
    mock_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    mock_scores = torch.zeros((1, 50000))
    
    should_stop = criteria(mock_input_ids, mock_scores)
    print(f"Should stop on repetition: {should_stop}")
    
    print("\nStopping criteria classes loaded successfully!")