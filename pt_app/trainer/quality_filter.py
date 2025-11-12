"""
Translation Quality Filter
Handles post-generation cleaning and validation for translation outputs
"""

import re
from typing import Optional, List, Tuple


class TranslationQualityFilter:
    """Filter and clean translation outputs"""
    
    def __init__(self, tokenizer, target_language='pt'):
        self.tokenizer = tokenizer
        self.target_language = target_language
        
        # Language detection keywords
        self.english_keywords = {
            'the', 'is', 'are', 'was', 'were', 'will', 'would', 'have', 'has', 'had',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where',
            'how', 'why', 'can', 'could', 'should', 'must', 'may', 'might',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours',
            'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        self.portuguese_keywords = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
            'é', 'são', 'foi', 'foram', 'será', 'serão', 'seria', 'seriam',
            'está', 'estão', 'estava', 'estavam', 'estará', 'estarão',
            'tem', 'têm', 'tinha', 'tinham', 'terá', 'terão',
            'eu', 'tu', 'você', 'ele', 'ela', 'nós', 'vocês', 'eles', 'elas',
            'me', 'te', 'se', 'nos', 'vos', 'lhe', 'lhes',
            'meu', 'minha', 'meus', 'minhas', 'teu', 'tua', 'seu', 'sua',
            'nosso', 'nossa', 'vosso', 'vossa',
            'e', 'ou', 'mas', 'porque', 'se', 'como', 'quando', 'onde',
            'que', 'qual', 'quais', 'quem', 'quanto', 'quantos',
            'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
            'por', 'pelo', 'pela', 'pelos', 'pelas', 'para', 'com', 'sem',
            'sobre', 'sob', 'entre', 'até', 'desde', 'durante', 'após',
            'antes', 'depois', 'mais', 'menos', 'muito', 'pouco', 'bem', 'mal'
        }
    
    def filter_translation(self, source: str, translation: str, verbose: bool = False) -> Optional[str]:
        """
        Main filtering pipeline
        
        Args:
            source: Source text
            translation: Generated translation
            verbose: Print filtering steps
            
        Returns:
            Cleaned translation or None if quality is too low
        """
        if not translation or not translation.strip():
            if verbose:
                print("  [FILTER] Empty translation")
            return None
        
        original = translation
        
        # Step 1: Extract target language content only
        translation = self._extract_target_language(translation)
        if verbose and translation != original:
            print(f"  [FILTER] Language mixing detected, extracted: {translation[:50]}...")
        
        if not translation or not translation.strip():
            if verbose:
                print("  [FILTER] No target language content found")
            return None
        
        # Step 2: Remove repetitions
        translation = self._remove_repetitions(translation)
        if verbose and len(translation) < len(original) * 0.8:
            print(f"  [FILTER] Removed repetitions")
        
        # Step 3: Length sanity check
        if not self._length_reasonable(source, translation):
            if verbose:
                print(f"  [FILTER] Length unreasonable (src: {len(source.split())} words, trans: {len(translation.split())} words)")
            return None
        
        # Step 4: Check for incomplete sentences
        if self._is_incomplete(translation):
            if verbose:
                print(f"  [FILTER] Incomplete sentence detected")
            return None
        
        # Step 5: Remove trailing artifacts
        translation = self._clean_artifacts(translation)
        
        return translation.strip()
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on keyword matching
        
        Returns:
            'en' for English, 'pt' for Portuguese, 'mixed' for mixed, 'unknown' otherwise
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 'unknown'
        
        en_count = sum(1 for w in words if w in self.english_keywords)
        pt_count = sum(1 for w in words if w in self.portuguese_keywords)
        
        # Calculate percentages
        total = len(words)
        en_pct = en_count / total if total > 0 else 0
        pt_pct = pt_count / total if total > 0 else 0
        
        # Decision logic
        if en_pct > 0.2 and pt_pct > 0.2:
            return 'mixed'
        elif en_pct > pt_pct and en_pct > 0.15:
            return 'en'
        elif pt_pct > en_pct and pt_pct > 0.15:
            return 'pt'
        else:
            return 'unknown'
    
    def _extract_target_language(self, text: str) -> str:
        """
        Extract only target language portions from mixed-language text
        """
        # Split by common delimiters
        segments = re.split(r'[\n]+|assistant\n|<\|.*?\|>', text)
        
        target_segments = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            lang = self._detect_language(segment)
            
            if lang == self.target_language:
                target_segments.append(segment)
            elif lang == 'mixed':
                # Try to extract Portuguese sentences
                sentences = re.split(r'[.!?]+', segment)
                for sent in sentences:
                    if self._detect_language(sent) == self.target_language:
                        target_segments.append(sent.strip())
        
        # If no target language found, return original (might be good translation)
        if not target_segments:
            # Check if original is target language
            if self._detect_language(text) in [self.target_language, 'unknown']:
                return text
            return ""
        
        return ' '.join(target_segments)
    
    def _remove_repetitions(self, text: str) -> str:
        """
        Remove duplicate sentences and repeated phrases
        """
        # Remove sentence-level duplicates
        sentences = re.split(r'([.!?]+)', text)
        
        cleaned_parts = []
        seen_sentences = set()
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            
            # Get punctuation if exists
            punct = sentences[i + 1].strip() if i + 1 < len(sentences) else ''
            
            if sentence:
                # Normalize for comparison
                normalized = re.sub(r'\s+', ' ', sentence.lower())
                
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    cleaned_parts.append(sentence)
                    if punct:
                        cleaned_parts.append(punct)
            
            i += 2 if punct else 1
        
        text = ''.join(cleaned_parts)
        
        # Remove repeated n-grams (3+ words repeated)
        words = text.split()
        if len(words) > 6:
            # Check for 3-gram repetitions
            for ngram_size in [5, 4, 3]:
                cleaned_words = []
                i = 0
                while i < len(words):
                    # Check if this n-gram repeats immediately after
                    if i + ngram_size * 2 <= len(words):
                        ngram1 = ' '.join(words[i:i+ngram_size])
                        ngram2 = ' '.join(words[i+ngram_size:i+ngram_size*2])
                        
                        if ngram1.lower() == ngram2.lower():
                            # Skip the repeated n-gram
                            cleaned_words.extend(words[i:i+ngram_size])
                            i += ngram_size * 2
                            continue
                    
                    cleaned_words.append(words[i])
                    i += 1
                
                words = cleaned_words
        
        return ' '.join(words)
    
    def _length_reasonable(self, source: str, translation: str, 
                          min_ratio: float = 0.3, max_ratio: float = 2.5) -> bool:
        """
        Check if translation length is reasonable compared to source
        
        Args:
            source: Source text
            translation: Translation text
            min_ratio: Minimum length ratio (translation/source)
            max_ratio: Maximum length ratio (translation/source)
        """
        source_words = len(source.split())
        trans_words = len(translation.split())
        
        if source_words == 0:
            return trans_words > 0
        
        ratio = trans_words / source_words
        
        return min_ratio <= ratio <= max_ratio
    
    def _is_incomplete(self, text: str) -> bool:
        """
        Check if translation appears incomplete or cut off
        """
        text = text.strip()
        
        if not text:
            return True
        
        # Should end with proper punctuation
        if not re.search(r'[.!?"\']$', text):
            # Exception: very short phrases might not need punctuation
            if len(text.split()) > 3:
                return True
        
        # Check for incomplete sentence patterns
        incomplete_patterns = [
            r'\b(que|de|para|com|em)$',  # Ends with preposition
            r'\b(o|a|os|as|um|uma)$',     # Ends with article
            r',\s*$',                      # Ends with comma
            r'\b(muito|pouco|mais|menos)$',  # Ends with incomplete modifier
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _clean_artifacts(self, text: str) -> str:
        """
        Remove common artifacts from generation
        """
        # Remove special tokens that might leak through
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\bassistant\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\buser\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsystem\b', '', text, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        
        return text.strip()
    
    def get_statistics(self, translations: List[Tuple[str, str, str]]) -> dict:
        """
        Get filtering statistics
        
        Args:
            translations: List of (source, raw_translation, filtered_translation) tuples
            
        Returns:
            Dictionary with statistics
        """
        total = len(translations)
        filtered_out = sum(1 for _, _, filtered in translations if filtered is None)
        
        language_mixing = 0
        repetitions = 0
        length_issues = 0
        incomplete = 0
        
        for source, raw, filtered in translations:
            if filtered is None:
                if not self._extract_target_language(raw):
                    language_mixing += 1
                if not self._length_reasonable(source, raw):
                    length_issues += 1
                if self._is_incomplete(raw):
                    incomplete += 1
            else:
                if len(filtered) < len(raw) * 0.9:
                    repetitions += 1
        
        return {
            'total': total,
            'filtered_out': filtered_out,
            'pass_rate': (total - filtered_out) / total if total > 0 else 0,
            'language_mixing': language_mixing,
            'repetitions': repetitions,
            'length_issues': length_issues,
            'incomplete': incomplete
        }


if __name__ == "__main__":
    # Test the filter
    class DummyTokenizer:
        pass
    
    filter = TranslationQualityFilter(DummyTokenizer())
    
    test_cases = [
        # (source, translation, expected_result)
        ("Hello", "Olá", "should_pass"),
        ("Hello", "Olá Olá Olá", "should_remove_repetition"),
        ("Hello", "Hello Olá", "should_extract_portuguese"),
        ("Hello", "Olá assistant\nOlá", "should_clean"),
        ("Hello", "O a", "should_fail_incomplete"),
        ("Hello", "Olá! Olá! Olá!", "should_remove_repetition"),
        ("Short text", "Este é um texto muito muito muito muito longo que não faz sentido para o tamanho da entrada", "should_fail_length"),
    ]
    
    print("Testing TranslationQualityFilter:\n")
    for source, translation, expected in test_cases:
        result = filter.filter_translation(source, translation, verbose=True)
        print(f"Source: {source}")
        print(f"Input: {translation}")
        print(f"Output: {result}")
        print(f"Expected: {expected}")
        print("-" * 50)