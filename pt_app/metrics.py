# mt_metrics.py

import sacrebleu
from rouge_score import rouge_scorer
import evaluate
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
import mlx.core as mx

from mlx_lm.tuner.callbacks import TrainingCallback
from pt_app.trainer.mlx_utils import mlx_generate, mlx_generate_simple


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
                 eval_samples: int = 50,
                 prompt_template: str = "Translate to Portuguese: {source}\n\nPortuguese:"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = mt_evaluator
        self.log_dir = log_dir
        self.eval_samples = eval_samples
        self.prompt_template = prompt_template
        self.metrics_history = []
        
        os.makedirs(log_dir, exist_ok=True)
        
    def on_step_end(self, step, **kwargs):
        """Debug callback system"""
        if step % 10 == 0:  # Same frequency as steps_per_eval
            print(f"[DEBUG] on_step_end called at step {step}")
        
    def on_eval_begin(self, **kwargs):
        """Debug evaluation start"""
        print(f"[DEBUG] on_eval_begin called with kwargs: {list(kwargs.keys())}")
    
    def on_eval_end(self, **kwargs):
        """Compute MT metrics after validation"""
        
        # Get validation dataset from kwargs
        val_dataset = kwargs.get('val_dataset', None)
        val_loss = kwargs.get('val_loss', 0.0)
        step = kwargs.get('step', 0)
        
        if val_dataset is None:
            print("[DEBUG] val_dataset is None, returning early")
            return
        
        print(f"\n[MT METRICS] Computing translation metrics at step {step}...")
        print(f"[DEBUG] val_dataset length: {len(val_dataset)}")
        
        sources = []
        references = []
        predictions = []
        
        # Sample subset for evaluation
        num_eval = min(self.eval_samples, len(val_dataset))
        
        for i in range(num_eval):
            try:
                # Decode the tokenized sample
                token_ids = val_dataset[i]
                full_text = self.tokenizer.decode(token_ids)
                
                # Extract English and Portuguese from the full text
                en_pattern = "'en': "
                en_idx = full_text.find(en_pattern)
                if en_idx == -1:
                    en_pattern = '"en": '
                    en_idx = full_text.find(en_pattern)
                    
                pt_pattern = "'pt': "
                pt_idx = full_text.find(pt_pattern)
                if pt_idx == -1:
                    pt_pattern = '"pt": '
                    pt_idx = full_text.find(pt_pattern)
                
                if en_idx != -1 and pt_idx != -1:
                    # Extract source (English) text
                    quote_start = full_text.find('"', en_idx + len(en_pattern))
                    if quote_start == -1:
                        quote_start = full_text.find("'", en_idx + len(en_pattern))
                        
                    quote_end = full_text.find('"', quote_start + 1)
                    if quote_end == -1:
                        quote_end = full_text.find("'", quote_start + 1)
                    
                    source_text = full_text[quote_start+1:quote_end]
                    
                    # Extract target (Portuguese) text
                    quote_start = full_text.find("'", pt_idx + len(pt_pattern))
                    if quote_start == -1:
                        quote_start = full_text.find('"', pt_idx + len(pt_pattern))
                        
                    quote_end = full_text.find("'", quote_start + 1)
                    if quote_end == -1:
                        quote_end = full_text.find('"', quote_start + 1)
                    
                    target_text = full_text[quote_start+1:quote_end]
                    
                    print(f"[DEBUG] Extracted from sample {i}:")
                    print(f"  Source: {source_text[:50]}...")
                    print(f"  Target: {target_text[:50]}...")
                    
                    # Create prompt
                    prompt = self.prompt_template.format(source=source_text)
                    
                    try:
                        # Generate using the function defined above
                        # full_output = mlx_generate(
                        #     model=self.model,
                        #     tokenizer=self.tokenizer,
                        #     prompt=prompt,
                        #     max_tokens=150,
                        #     temp=0.1,
                        #     top_p=0.9,
                        #     repeat_penalty=1.1
                        # )
                        full_output = mlx_generate_simple(
                                    model=self.model,
                                    tokenizer=self.tokenizer,
                                    prompt=prompt,
                                    max_tokens=150
                                )
                        translation = full_output.replace(prompt, "").strip()
                        
                        sources.append(source_text)
                        references.append(target_text)
                        predictions.append(translation)
                        
                        print(f"Sample {i}:")
                        print(f"  Source: {source_text[:50]}...")
                        print(f"  Target: {target_text[:50]}...")
                        print(f"  Predicted: {translation[:50]}...")
                        
                    except Exception as e:
                        print(f"[WARNING] Error generating translation: {e}")
                        import traceback
                        traceback.print_exc()
            
            except Exception as e:
                    print(f"[WARNING] Error processing sample: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Compute metrics if we have predictions
        if predictions:
            try:
                metrics = self.evaluator.compute_metrics(predictions, references)
                print(self.evaluator.format_metrics(metrics))
                
                # Save metrics history
                self.metrics_history.append({
                    'step': step,
                    'val_loss': float(val_loss) if val_loss else 0.0,
                    'num_samples': len(predictions),
                    'metrics': {
                        'bleu': metrics.bleu,
                        'rouge_l': metrics.rouge_l,
                        'chrf': metrics.chrf,
                        'ter': metrics.ter,
                        'bertscore': metrics.bertscore
                    }
                })
                
                # Save metrics to file
                metrics_file = os.path.join(self.log_dir, "mt_metrics_history.json")
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
                
                # Save sample translations
                samples_file = os.path.join(self.log_dir, f"translations_step_{step}.json")
                samples_to_save = []
                for i in range(min(10, len(predictions))):
                    samples_to_save.append({
                        'source_en': sources[i],
                        'reference_pt': references[i],
                        'predicted_pt': predictions[i]
                    })
                
                with open(samples_file, 'w') as f:
                    json.dump(samples_to_save, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"[WARNING] Error computing metrics: {e}")
        else:
            print("[WARNING] No predictions generated, skipping metrics calculation")