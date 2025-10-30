import sacrebleu
from rouge_score import rouge_scorer
import evaluate
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import os
from datetime import datetime
import mlx.core as mx
import re
import ast

from mlx_lm.tuner.callbacks import TrainingCallback
from pt_app.trainer.mlx_utils import *


@dataclass
class MTMetrics:
    """Container for machine translation metrics"""
    bleu: float
    rouge_l: float
    chrf: float
    ter: float = None
    bertscore: Dict[str, float] = None


class MTEvaluator:
    """Machine Translation evaluation metrics for EN->PT"""
    
    def __init__(self, source_lang="en", target_lang="pt", use_bert_score=False):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Initialize metrics
        self.bleu = sacrebleu.BLEU()
        self.chrf = sacrebleu.CHRF()
        self.ter = sacrebleu.TER()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        self.bertscore = None
        if use_bert_score:
            try:
                self.bertscore = evaluate.load("bertscore")
            except:
                print("[WARNING] BERTScore not available")
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> MTMetrics:
        """Compute all translation metrics"""
        
        if not predictions or not references:
            return MTMetrics(bleu=0.0, rouge_l=0.0, chrf=0.0, ter=100.0)
        
        # BLEU score
        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        
        # chrF score - good for Portuguese
        chrf_score = self.chrf.corpus_score(predictions, [references]).score
        
        # TER score (lower is better)
        ter_score = self.ter.corpus_score(predictions, [references]).score
        
        # ROUGE-L score
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores.append(scores['rougeL'].fmeasure)
        rouge_l = np.mean(rouge_scores) * 100
        
        # BERTScore
        bertscore_dict = None
        if self.bertscore and len(predictions) > 0:
            try:
                results = self.bertscore.compute(
                    predictions=predictions, 
                    references=references, 
                    lang=self.target_lang
                )
                bertscore_dict = {
                    'precision': np.mean(results['precision']) * 100,
                    'recall': np.mean(results['recall']) * 100,
                    'f1': np.mean(results['f1']) * 100
                }
            except Exception as e:
                print(f"[WARNING] BERTScore computation failed: {e}")
        
        return MTMetrics(
            bleu=bleu_score,
            rouge_l=rouge_l,
            chrf=chrf_score,
            ter=ter_score,
            bertscore=bertscore_dict
        )
    
    def format_metrics(self, metrics: MTMetrics) -> str:
        """Format metrics for display"""
        output = f"""
EN→PT Translation Metrics:
  BLEU:    {metrics.bleu:.2f}
  ROUGE-L: {metrics.rouge_l:.2f}  
  chrF:    {metrics.chrf:.2f}  (good for PT morphology)
  TER:     {metrics.ter:.2f}  (lower is better)"""
        
        if metrics.bertscore:
            output += f"""
  BERTScore F1: {metrics.bertscore['f1']:.2f}"""
        
        return output


class MTTrainingCallback(TrainingCallback):
    """Training callback that computes MT metrics during validation"""
    
    def __init__(self, 
                 model,
                 tokenizer,
                 mt_evaluator: MTEvaluator,
                 log_dir: str = "./logs",
                 eval_samples=8,
                 train_dataset_size=None,
                 batch_size=None,
                 prompt_template: str = "Translate to Portuguese: {source}\n\nPortuguese:",
                 # Add this parameter to pass in the actual source/target pairs
                 validation_pairs: List[Dict[str, str]] = None):
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = mt_evaluator
        self.log_dir = log_dir
        self.eval_samples = eval_samples
        self.train_dataset_size = train_dataset_size
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.metrics_history = []
        
        # Store the validation pairs directly
        self.validation_pairs = validation_pairs
        
        if train_dataset_size and batch_size:
            self.steps_per_epoch = max(1, train_dataset_size // batch_size)
            print(f"[INFO] MT Callback: {train_dataset_size} samples, {batch_size} batch size")
            print(f"[INFO] Estimated steps per epoch: {self.steps_per_epoch}")
        
        os.makedirs(log_dir, exist_ok=True)
        
    def on_step_end(self, step, **kwargs):
        """Debug callback system"""
        if step % 10 == 0:
            print(f"[DEBUG] on_step_end called at step {step}")
    
    def on_eval_begin(self, **kwargs):
        """Debug evaluation start"""
        print(f"[DEBUG] on_eval_begin called with kwargs: {list(kwargs.keys())}")
    
    def on_eval_end(self, **kwargs):
        """Compute MT metrics after validation"""
        val_loss = kwargs.get('val_loss', 0.0)
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('iters', 0)
        
        # If we don't have validation pairs, try to extract from the dataset
        if self.validation_pairs is None:
            print("[WARNING] No validation pairs provided to callback. Attempting extraction...")
            val_dataset = kwargs.get('val_dataset', None)
            if val_dataset is None:
                print("[ERROR] No validation dataset available")
                return
            
            # Try to extract pairs from dataset
            self.validation_pairs = []
            for i in range(min(20, len(val_dataset))):
                token_ids = val_dataset[i]
                full_text = self.tokenizer.decode(token_ids)
                
                # Simple extraction - look for the dictionary pattern
                match = re.search(r"\{'en':\s*'([^']*)',\s*'pt':\s*'([^']*)'\}", full_text)
                if not match:
                    match = re.search(r'\{"en":\s*"([^"]*)",\s*"pt":\s*"([^"]*)"\}', full_text)
                
                if match:
                    self.validation_pairs.append({
                        'en': match.group(1),
                        'pt': match.group(2)
                    })
            
            if not self.validation_pairs:
                print("[ERROR] Could not extract any validation pairs")
                return
            
            print(f"[INFO] Extracted {len(self.validation_pairs)} validation pairs")
        
        # Calculate steps per epoch dynamically
        if hasattr(self, 'train_dataset_size') and hasattr(self, 'batch_size'):
            steps_per_epoch = self.train_dataset_size // self.batch_size
        else:
            steps_per_epoch = 300
        
        # Determine if this is a milestone step
        is_first_or_last = (step <= 5 or step >= total_steps - 5)
        is_epoch_boundary = (step % steps_per_epoch < 5)
        
        # Compute metrics less frequently
        metrics_interval = max(steps_per_epoch // 2, 50)
        compute_metrics = (is_first_or_last or is_epoch_boundary or step % metrics_interval == 0)
        
        if not compute_metrics:
            self.metrics_history.append({
                'step': step,
                'val_loss': float(val_loss),
                'metrics_computed': False
            })
            print(f"\n[INFO] Skipping full metrics at step {step}, recording validation loss only")
            return
        
        print(f"\n[MT METRICS] Computing translation metrics at step {step}...")
        
        # Use the stored validation pairs
        num_eval = min(self.eval_samples, len(self.validation_pairs))
        
        sources = []
        references = []
        predictions = []
        
        print(f"[INFO] Generating {num_eval} translations...")
        for i in range(num_eval):
            pair = self.validation_pairs[i]
            source_text = pair['en']
            target_text = pair['pt']
            
            sources.append(source_text)
            references.append(target_text)
            
            try:
                # Generate translation
                prompt = self.prompt_template.format(source=source_text)
                
                full_output = mlx_generate_optimized(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=75
                )
                
                # Extract translation
                translation = full_output.replace(prompt, "").strip()
                predictions.append(translation)
                
                if i < 3:  # Show first few
                    print(f"  Sample {i+1}:")
                    print(f"    Source: {source_text[:50]}...")
                    print(f"    Reference: {target_text[:50]}...")
                    print(f"    Generated: {translation[:50]}...")
                
            except Exception as e:
                print(f"[WARNING] Error generating translation {i+1}: {e}")
                predictions.append("")
        
        # Compute metrics
        if predictions and len(predictions) == len(references):
            try:
                metrics = self.evaluator.compute_metrics(predictions, references)
                print(self.evaluator.format_metrics(metrics))
                
                # Store metrics
                self.metrics_history.append({
                    'step': step,
                    'val_loss': float(val_loss),
                    'num_samples': len(predictions),
                    'metrics': {
                        'bleu': metrics.bleu,
                        'rouge_l': metrics.rouge_l,
                        'chrf': metrics.chrf,
                        'ter': metrics.ter,
                        'bertscore': metrics.bertscore
                    }
                })
                
                # Save metrics
                metrics_file = os.path.join(self.log_dir, "mt_metrics_history.json")
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
                
                print(f"[INFO] Metrics saved to {metrics_file}")
                
            except Exception as e:
                print(f"[ERROR] Failed to compute metrics: {e}")
        else:
            print(f"[ERROR] Failed to generate all translations")


# Helper function to prepare validation pairs from your dataset
def extract_validation_pairs_from_dataset(dataset, tokenizer, num_samples=20):
    """
    Extract validation pairs from a tokenized dataset.
    Handles both dict and tensor formats.
    """
    import re
    import ast
    
    validation_pairs = []
    
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        
        # Handle different dataset formats
        if isinstance(item, dict):
            # Dataset returns a dictionary - get the input_ids
            if 'input_ids' in item:
                token_ids = item['input_ids']
            elif 'tokens' in item:
                token_ids = item['tokens']
            else:
                # Try to find any key that looks like token ids
                for key in item.keys():
                    if 'id' in key.lower() or 'token' in key.lower():
                        token_ids = item[key]
                        break
                else:
                    print(f"[WARNING] Sample {i}: Could not find token ids in keys: {list(item.keys())}")
                    continue
        else:
            # Assume it's already token ids
            token_ids = item
        
        # Decode the tokens
        try:
            full_text = tokenizer.decode(token_ids)
        except Exception as e:
            print(f"[WARNING] Sample {i}: Failed to decode: {e}")
            continue
        
        # Try different extraction patterns for the dictionary
        extracted = False
        
        # Pattern 1: Simple regex for clean dictionaries
        patterns = [
            r"\{'en':\s*'([^']*)',\s*'pt':\s*'([^']*)'\}",
            r'\{"en":\s*"([^"]*)",\s*"pt":\s*"([^"]*)"\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text)
            if match:
                validation_pairs.append({
                    'en': match.group(1),
                    'pt': match.group(2)
                })
                extracted = True
                break
        
        # Pattern 2: Try ast.literal_eval for complex cases
        if not extracted:
            dict_start = full_text.find("{'en':")
            if dict_start == -1:
                dict_start = full_text.find('{"en":')
            
            if dict_start != -1:
                # Find the matching closing brace
                brace_count = 0
                idx = dict_start
                dict_end = -1
                
                for j in range(idx, len(full_text)):
                    if full_text[j] == '{':
                        brace_count += 1
                    elif full_text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            dict_end = j
                            break
                
                if dict_end != -1:
                    dict_str = full_text[dict_start:dict_end + 1]
                    try:
                        # Try to parse as Python dict
                        pair_dict = ast.literal_eval(dict_str)
                        if isinstance(pair_dict, dict) and 'en' in pair_dict and 'pt' in pair_dict:
                            validation_pairs.append({
                                'en': pair_dict['en'],
                                'pt': pair_dict['pt']
                            })
                            extracted = True
                    except:
                        pass
        
        if extracted and len(validation_pairs) <= 3:
            # Show first few extracted pairs
            pair = validation_pairs[-1]
            print(f"[INFO] Extracted pair {len(validation_pairs)}:")
            print(f"  EN: {pair['en'][:50]}...")
            print(f"  PT: {pair['pt'][:50]}...")
    
    print(f"[INFO] Successfully extracted {len(validation_pairs)} validation pairs")
    return validation_pairs