"""
Prompt templates for generating translation evaluation training data.
Each prompt generates 4 variations with different quality levels.
"""

# ============================================================================
# PROMPT: BALANCED (Original)
# Generates: Score 3-5, 6-7, 8-9, 10
# Use: General training, full quality spectrum
# ============================================================================

PROMPT_BALANCED = """You are a translation quality assessment expert for English → Portuguese translation.

Given this professional translation:
Source (English): {source_en}
Reference (Portuguese): {reference_pt}

Your task is to generate exactly **4 Portuguese translations** with controlled, explicit error types at different quality levels. Accept both European and Brazilian Portuguese variants.

CRITICAL GLOBAL RULES:
- All translations must preserve the core meaning of the English source
- Do NOT add new information or omit ideas unless the error type explicitly requires it
- Do NOT leave any words in English
- Keep sentence structure reasonably close to the original
- Each variation must contain ONLY the intended error(s)—no accidental mistakes
- Every item in "issues" must correspond to a VISIBLE error in the translation
- For GOOD and EXCELLENT translations, the issues list MUST be empty

SCORING GUIDELINES:
0-2: Incomprehensible or completely wrong meaning
3-4: ONE major error that significantly impacts comprehension
5-6: ONE major error with minor impact OR 2-3 minor errors combined
7-8: 1-2 minor errors, meaning fully preserved
9: Nearly perfect, one extremely subtle stylistic issue only
10: Perfect or equally valid alternative

REQUIRED VARIATIONS:

1. LOW QUALITY (Score 3-5):
Introduce EXACTLY ONE major error:
a) Wrong verb with semantic shift: "gostar"→"adorar" (like→love), "ir"→"vir" (go→come)
b) Missing required word: omit article ("o", "a") or preposition ("de", "para", "em")
c) Wrong word - same category: "cão"→"gato" (dog→cat), "carro"→"ônibus" (car→bus)
d) Diacritic changing meaning: "cafe"→"café", "esta"→"está", "e"→"é"
e) Wrong tense affecting meaning: present instead of past or vice versa
f) Gender/number disagreement: "o casa" (masc+fem), "os gato" (plural+singular)

Score 3-4: Error significantly impacts understanding
Score 5-6: Error noticeable but meaning still clear

2. MEDIUM QUALITY (Score 6-7):
Introduce 1-2 minor errors:
a) Unnecessary or missing article where both work: "o café" vs "café"
b) Missing non-critical diacritic: "voce"→"você", "nao"→"não"
c) Slightly unnatural word choice: correct but uncommon synonym
d) Minor preposition variation: "em casa" vs "na casa"
e) Missing contraction: "de o" instead of "do"

Choose 1 error for score 7, or 2 errors for score 6

3. GOOD QUALITY (Score 8-9):
Nearly perfect with EXACTLY ONE extremely subtle issue:
a) Slight formality mismatch for context
b) Word order acceptable but not optimal: "Sempre eu como" vs "Eu como sempre"
c) Overly literal phrasing: grammatically correct but native would phrase differently

Score 9: One subtle issue
Score 8: Issue slightly more noticeable
ALL grammar, vocabulary, diacritics must be correct

4. EXCELLENT QUALITY (Score 10):
Must be:
- Grammatically perfect, all diacritics correct
- Natural Portuguese (not word-for-word translation)
- Meaning fully preserved
- Different from reference in at least one way:
    * Different synonym ("adorar" vs "amar")
    * Different structure ("Eu adoro café" vs "Adoro café")
    * Different word order (if both valid)
- NO errors or issues whatsoever

IMPORTANT CONSTRAINTS:
- Do not combine multiple major errors in one translation
- If introducing a diacritic error, choose words where it matters: café≠cafe, está≠esta
- Each variation must have CLEARLY DIFFERENT error types
- Be specific about which error you introduced in the error_type field

OUTPUT FORMAT:
For EACH translation, return:
{{
  "translation": "Portuguese text",
  "score": exact_number_0_to_10,
  "error_type": "specific_error_introduced (or 'none' for score 10)",
  "issues": ["specific", "observable", "problems"],
  "feedback": "1-2 sentences explaining the score and error"
}}

Return ONLY a valid JSON array with 4 variations, ordered lowest to highest score.
"""


# ============================================================================
# PROMPT: LOW SCORES
# Generates: Score 1-2, 2-3, 3-4, 4-5
# Use: Training to recognize BAD translations, fix over-optimistic scoring
# ============================================================================

PROMPT_LOW_SCORES = """You are a translation quality assessment expert for English → Portuguese translation.

Given this professional translation:
Source (English): {source_en}
Reference (Portuguese): {reference_pt}

Your task is to generate exactly **4 BAD Portuguese translations** with major errors at LOW quality levels (scores 1-5). 
This is for training a model to recognize what BAD translations look like. Accept both European and Brazilian Portuguese variants.

CRITICAL: Focus on MAJOR errors that significantly impact meaning or comprehension.

SCORING GUIDELINES:
0-1: Completely incomprehensible, wrong language, or opposite meaning
2-3: Multiple major errors OR one critical semantic error
3-4: One major error that significantly impacts comprehension
4-5: One major error with moderate impact

REQUIRED VARIATIONS:

1. VERY BAD (Score 1-2):
Multiple major errors OR critical semantic failure:
a) Wrong meaning entirely: "I love cats" → "Eu odeio gatos" (I hate cats)
b) Mixed languages: Keep 2-3 English words untranslated
c) Wrong verb completely changing action: "eat" → "drink", "buy" → "sell"
d) Opposite words: "big" → "small", "hot" → "cold"
e) Gibberish or random words that don't form meaning
f) Critical grammar making it incomprehensible

2. BAD (Score 2-3):
TWO major errors affecting meaning:
a) Wrong verb + wrong gender: "ele comeu" → "ela bebeu"
b) Missing verb + wrong preposition
c) Wrong tense + semantic error
d) Multiple semantic shifts
e) Wrong subject pronoun + wrong noun

3. POOR (Score 3-4):
ONE major error significantly impacting understanding:
a) Critical semantic error: "finger" → "toe", "car" → "bus", "won" → "lost"
b) Verb changing core meaning: "resigned" → "promoted", "forgot" → "remembered"
c) Missing critical word: omit main verb or subject
d) Wrong gender/number causing confusion: "o casa", "as gato"
e) Major tense error: future → past changes event timeline
f) Wrong preposition changing location/direction entirely

4. MEDIOCRE (Score 4-5):
ONE major error with moderate impact:
a) Wrong word (same category): "dog" → "cat", "Monday" → "Friday"  
b) Missing article/preposition causing ambiguity
c) Wrong verb with partial meaning overlap
d) Significant tense mismatch
e) Missing diacritic changing word meaning: "esta" (this) vs "está" (is)

IMPORTANT:
- These should be clearly BAD translations
- Model needs to learn the difference between bad (1-5) and acceptable (6-10)
- Be creative with major semantic errors
- Each variation must have clearly different error types
- Issues list must explicitly describe what's wrong

OUTPUT FORMAT:
For EACH translation, return:
{{
  "translation": "Portuguese text",
  "score": exact_number_1_to_5,
  "error_type": "specific_major_error_introduced",
  "issues": ["specific", "major", "problems"],
  "feedback": "2-3 sentences explaining why this is a bad translation"
}}

Return ONLY a valid JSON array with 4 bad variations, ordered lowest to highest score.
"""


# ============================================================================
# PROMPT: MIDDLE RANGE
# Generates: Score 4-5, 5-6, 6-7, 7-8
# Use: Training on ambiguous cases, boundary between acceptable/unacceptable
# ============================================================================

PROMPT_MIDDLE_RANGE = """You are a translation quality assessment expert for English → Portuguese translation.

Given this professional translation:
Source (English): {source_en}
Reference (Portuguese): {reference_pt}

Your task is to generate exactly **4 Portuguese translations** in the MIDDLE quality range (scores 4-8).
This teaches the model to distinguish between "mediocre", "acceptable", and "good" translations.
Accept both European and Brazilian Portuguese variants.

CRITICAL: Focus on the BOUNDARY between unacceptable (≤5) and acceptable (≥6).

SCORING GUIDELINES:
4-5: Major error with moderate impact, translation is poor but comprehensible
5-6: Multiple minor errors OR one noticeable error, borderline acceptable
6-7: 2-3 minor errors, acceptable but flawed
7-8: 1-2 minor errors, good translation with small issues

REQUIRED VARIATIONS:

1. POOR/MEDIOCRE (Score 4-5):
ONE major error that's noticeable but doesn't destroy meaning:
a) Wrong word (same semantic field): "carro" → "caminhão" (car → truck)
b) Verb tense error affecting clarity: "vai fazer" → "fez" (future → past)
c) Missing important preposition: "morar em Lisboa" → "morar Lisboa"
d) Gender/number error on important noun: "as livros" (plural article + singular noun)
e) Wrong collocation: "fazer uma decisão" instead of "tomar uma decisão"
f) Unnatural structure causing confusion

Score 4: Error significantly impacts flow/understanding
Score 5: Error noticeable but message still gets across

2. BORDERLINE (Score 5-6):
TWO minor errors OR one moderate error:
a) Missing article + wrong preposition
b) Diacritic error + slight word choice issue
c) Unnatural word order + missing contraction
d) Slightly wrong verb + formality mismatch
e) Acceptable but uncommon phrasing + minor grammar issue

Score 5: Errors together create noticeable problem
Score 6: Errors minor enough that translation still acceptable

3. ACCEPTABLE (Score 6-7):
2-3 minor errors that don't significantly affect meaning:
a) Missing non-critical articles: "Eu gosto de café" → "Eu gosto café"
b) Minor preposition variations: "em" vs "na"
c) Slight formality issues + missing diacritic
d) Unnatural but grammatically correct phrasing
e) Word choice slightly off but meaning clear

Score 6: Multiple small issues add up
Score 7: Issues are minimal and don't distract

4. GOOD (Score 7-8):
1-2 very minor issues:
a) Slight formality mismatch for context
b) Word order acceptable but not optimal
c) One missing diacritic on non-critical word
d) Slightly literal translation (correct but not natural)
e) Acceptable synonym that's less common

Score 7: Issue noticeable to careful reader
Score 8: Issue barely noticeable, translation flows well

IMPORTANT:
- Focus on GRADATIONS in the middle range
- Model needs to learn: when does "flawed" become "unacceptable"?
- Vary error combinations to show different paths to same score
- Be precise about WHY each score was chosen
- Show that multiple minor errors ≠ one major error

OUTPUT FORMAT:
For EACH translation, return:
{{
  "translation": "Portuguese text",
  "score": exact_number_4_to_8,
  "error_type": "specific_error(s)_introduced",
  "issues": ["specific", "observable", "problems"],
  "feedback": "2-3 sentences explaining the score and why it falls in this quality range"
}}

Return ONLY a valid JSON array with 4 variations, ordered lowest to highest score.
"""
