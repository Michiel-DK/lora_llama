from datasets import Dataset, DatasetDict

# Your current data (assuming it's in a pandas DataFrame or similar)
# Based on your example, you have English (En) and Portuguese (Pt) columns

# Create the dataset in OPUS Books format
def create_opus_format(en_texts, pt_texts, ids=None):
    """
    Convert your En-Pt parallel data to OPUS Books format
    
    Args:
        en_texts: List of English texts
        pt_texts: List of Portuguese texts
        ids: Optional list of IDs (will generate if not provided)
    """
    if ids is None:
        ids = [str(i) for i in range(len(en_texts))]
    
    # OPUS Books format uses a 'translation' field with language codes as keys
    data = {
        'id': ids,
        'translation': [
            {'en': en, 'pt': pt} 
            for en, pt in zip(en_texts, pt_texts)
        ]
    }
    
    return Dataset.from_dict(data)