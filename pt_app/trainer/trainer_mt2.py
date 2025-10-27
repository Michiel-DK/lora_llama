from mlx_lm_lora.utils import from_pretrained
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm_lora.trainer.datasets import CacheDataset
import mlx.optimizers as optim
import mlx.core as mx

# MT metrics imports
import sacrebleu
from rouge_score import rouge_scorer
import evaluate
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from params import *
import math
from datetime import datetime
import os


@dataclass
class MTMetrics:
    """Container for machine translation metrics"""
    bleu: float
    rouge_l: float
    chrf: float
    ter: float = None
    bertscore: Dict[str, float] = None


class LloraTrainer():
    
    def __init__(self, enable_mt_metrics=True, source_lang="en", target_lang="pt"):
        
        self.model_name = MODEL_NAME
        self.lora_config = LORA_CONFIG
        self.quantization_config = QUANTIZATION_CONFIG
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.max_seq_length = MAX_SEQ_LENGTH
        self.optimizer_config = OPTIMIZER_CONFIG
        self.training_args = TRAINING_ARGS
        self.adapter_path = ADAPTER_PATH
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # MT metrics setup
        self.enable_mt_metrics = enable_mt_metrics
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.metrics_history = []
        
        if self.enable_mt_metrics:
            self._init_mt_evaluators()
    
    def _init_mt_evaluators(self):
        """Initialize MT evaluation metrics"""
        print("[INFO] Initializing MT metrics...")
        self.bleu = sacrebleu.BLEU()
        self.chrf = sacrebleu.CHRF()
        self.ter = sacrebleu.TER()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Optional BERTScore
        try:
            self.bertscore = evaluate.load("bertscore")
            print("[INFO] BERTScore loaded successfully")
        except:
            self.bertscore = None
            print("[INFO] BERTScore not available - continuing without it")
    
    def get_model(self):
        self.model, self.tokenizer = from_pretrained(
            model=self.model_name,
            lora_config=self.lora_config,
            quantized_load=self.quantization_config,
        )
        print_trainable_parameters(self.model)
        
        print(f"\n[CONFIG] Model: {self.model_name}")
        print(f"[CONFIG] LoRA rank: {self.lora_config['rank']}")
        print(f"[CONFIG] Quantization: {self.quantization_config['bits']}-bit")
        if self.enable_mt_metrics:
            print(f"[CONFIG] MT Metrics: {self.source_lang} → {self.target_lang}")
        
        return self.model, self.tokenizer
    
    def get_optimizer(self):
        self.opt = optim.AdamW(
            learning_rate=self.optimizer_config["learning_rate"],
            weight_decay=self.optimizer_config.get("weight_decay", 0.01),
            betas=[self.optimizer_config.get("beta1", 0.9), 
                   self.optimizer_config.get("beta2", 0.999)],
            eps=self.optimizer_config.get("eps", 1e-8)
        )
        return self.opt
    
    def compute_mt_metrics(self, predictions: List[str], references: List[str]) -> MTMetrics:
        """Compute all translation metrics"""
        
        if not predictions or not references:
            return MTMetrics(bleu=0.0, rouge_l=0.0, chrf=0.0, ter=100.0)
        
        # BLEU score
        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        
        # chrF score - particularly good for Portuguese
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
        if self.bertscore:
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
    
    def evaluate_translations(self, val_dataset, num_samples=50, prompt_template="Translate to Portuguese: {source}\n\nPortuguese:"):
        """Generate translations and compute metrics"""
        
        predictions = []
        references = []
        sources = []
        
        num_eval = min(num_samples, len(val_dataset))
        print(f"\n[MT EVAL] Generating {num_eval} translations for evaluation...")
        
        for i in range(num_eval):
            try:
                sample = val_dataset[i]
                
                # Debug: print sample structure
                if i == 0:
                    print(f"[DEBUG] Sample type: {type(sample)}")
                    if isinstance(sample, dict):
                        print(f"[DEBUG] Sample keys: {sample.keys()}")
                
                # Extract source and target - adjust based on your dataset
                if isinstance(sample, dict):
                    source_text = sample.get('source', sample.get('en', sample.get('text', '')))
                    target_text = sample.get('target', sample.get('pt', ''))
                    
                    # If no clear source/target, try to parse from text field
                    if not source_text and 'text' in sample:
                        # Assuming format like "English: ... Portuguese: ..."
                        text = sample['text']
                        if "Portuguese:" in text:
                            parts = text.split("Portuguese:")
                            source_text = parts[0].replace("English:", "").strip()
                            target_text = parts[1].strip() if len(parts) > 1 else ""
                else:
                    continue
                
                if not source_text or not target_text:
                    continue
                
                # Create prompt
                prompt = prompt_template.format(source=source_text)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="np",
                    max_length=self.max_seq_length,
                    truncation=True
                )
                
                # Generate translation
                with mx.no_grad():
                    output_ids = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=150,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and extract translation
                full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                translation = full_output.replace(prompt, "").strip()
                
                predictions.append(translation)
                references.append(target_text)
                sources.append(source_text)
                
                # Print first example
                if i == 0:
                    print(f"\n[MT EVAL] Example translation:")
                    print(f"  Source: {source_text[:100]}...")
                    print(f"  Reference: {target_text[:100]}...")
                    print(f"  Predicted: {translation[:100]}...")
                
            except Exception as e:
                print(f"[WARNING] Error processing sample {i}: {e}")
                continue
        
        print(f"[MT EVAL] Successfully processed {len(predictions)} translations")
        
        # Compute metrics
        if predictions:
            metrics = self.compute_mt_metrics(predictions, references)
            return metrics, predictions, references, sources
        else:
            return None, [], [], []
    
    def train(self, train_set, val_set):
        
        num_samples = len(train_set)
        batches_per_epoch = math.ceil(num_samples / BATCH_SIZE)
        iters = EPOCHS * batches_per_epoch
        print(f"[INFO] Calculated {iters} iterations from {EPOCHS} epochs (dataset size: {num_samples}, batch size: {self.batch_size})")
        
        timing = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        new_adapter_path = os.path.join(ADAPTER_PATH, f'{timing}_epoch{iters}.safetensors')
        log_dir = os.path.join("./logs", f"training_{timing}")
        os.makedirs(ADAPTER_PATH, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Store references for callback
        self._val_dataset = val_set
        self._log_dir = log_dir
        self._current_step = 0
        
        # Create custom callback with MT metrics
        callback = self._create_training_callback()
        
        sft_args = SFTTrainingArgs(
            batch_size=self.batch_size,
            iters=iters,
            val_batches=self.training_args["val_batches"],
            steps_per_report=self.training_args["steps_per_report"],
            steps_per_eval=self.training_args["steps_per_eval"],
            steps_per_save=self.training_args["steps_per_save"],
            adapter_file=new_adapter_path,
            max_seq_length=self.max_seq_length,
            grad_checkpoint=self.training_args["grad_checkpoint"],
        )
        
        print("[INFO] Training with MT metrics enabled" if self.enable_mt_metrics else "[INFO] Training without MT metrics")
        
        train_sft(
            model=self.model,
            args=sft_args,
            optimizer=self.opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(val_set),
            training_callback=callback
        )
        
        # Print final metrics summary
        if self.enable_mt_metrics and self.metrics_history:
            self._print_final_metrics()
        
        return new_adapter_path
    
    def _create_training_callback(self):
        """Create training callback with MT metrics"""
        
        trainer_instance = self
        
        class MTTrainingCallback(TrainingCallback):
            
            def on_train_begin(self, model, train_args, **kwargs):
                """Called at the beginning of training"""
                print("[CALLBACK] Training started - MT metrics will be computed every eval step")
                super().on_train_begin(model, train_args, **kwargs)
            
            def on_eval_end(self, model, val_loss, val_dataset, **kwargs):
                """Compute MT metrics after each evaluation"""
                
                print(f"\n[CALLBACK] on_eval_end called with val_loss={val_loss}")
                
                if not trainer_instance.enable_mt_metrics:
                    return
                
                # Get current step
                step = kwargs.get('step', trainer_instance._current_step)
                trainer_instance._current_step = step
                
                try:
                    # Evaluate translations using stored val_dataset
                    metrics, predictions, references, sources = trainer_instance.evaluate_translations(
                        trainer_instance._val_dataset,  # Use stored dataset
                        num_samples=min(50, len(trainer_instance._val_dataset))
                    )
                    
                    if metrics:
                        # Print metrics
                        print(f"\n{'='*60}")
                        print(f"Step {step} - MT Metrics Evaluation")
                        print(f"{'='*60}")
                        print(f"Validation Loss: {val_loss:.4f}")
                        print(f"{trainer_instance.source_lang.upper()}→{trainer_instance.target_lang.upper()} Translation Metrics:")
                        print(f"  BLEU:    {metrics.bleu:.2f}")
                        print(f"  chrF:    {metrics.chrf:.2f}  (good for {trainer_instance.target_lang.upper()})")
                        print(f"  ROUGE-L: {metrics.rouge_l:.2f}")
                        print(f"  TER:     {metrics.ter:.2f}  (lower is better)")
                        if metrics.bertscore:
                            print(f"  BERTScore F1: {metrics.bertscore['f1']:.2f}")
                        print(f"{'='*60}\n")
                        
                        # Save metrics
                        trainer_instance.metrics_history.append({
                            'step': step,
                            'val_loss': float(val_loss) if val_loss else 0.0,
                            'metrics': {
                                'bleu': metrics.bleu,
                                'rouge_l': metrics.rouge_l,
                                'chrf': metrics.chrf,
                                'ter': metrics.ter,
                                'bertscore': metrics.bertscore
                            }
                        })
                        
                        # Save metrics to file
                        metrics_file = os.path.join(trainer_instance._log_dir, "metrics_history.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(trainer_instance.metrics_history, f, indent=2)
                        
                        # Save sample translations
                        if predictions:
                            samples_file = os.path.join(trainer_instance._log_dir, f"translations_step_{step}.json")
                            samples = []
                            for i in range(min(10, len(predictions))):
                                samples.append({
                                    f'source_{trainer_instance.source_lang}': sources[i],
                                    f'reference_{trainer_instance.target_lang}': references[i],
                                    f'predicted_{trainer_instance.target_lang}': predictions[i]
                                })
                            
                            with open(samples_file, 'w') as f:
                                json.dump(samples, f, indent=2, ensure_ascii=False)
                    else:
                        print("[WARNING] No translations generated for metrics computation")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to compute MT metrics: {e}")
                    import traceback
                    traceback.print_exc()
                
                super().on_eval_end(model, val_loss, val_dataset, **kwargs)
        
        return MTTrainingCallback()
    
    def _print_final_metrics(self):
        """Print summary of best metrics achieved during training"""
        
        if not self.metrics_history:
            print("[INFO] No metrics history to summarize")
            return
        
        best_bleu = max(self.metrics_history, key=lambda x: x['metrics']['bleu'])
        best_chrf = max(self.metrics_history, key=lambda x: x['metrics']['chrf'])
        lowest_ter = min(self.metrics_history, key=lambda x: x['metrics']['ter'])
        
        print(f"\n{'='*60}")
        print(f"FINAL METRICS SUMMARY ({self.source_lang.upper()}→{self.target_lang.upper()})")
        print(f"{'='*60}")
        print(f"Best BLEU:  {best_bleu['metrics']['bleu']:.2f} (step {best_bleu['step']})")
        print(f"Best chrF:  {best_chrf['metrics']['chrf']:.2f} (step {best_chrf['step']})")
        print(f"Lowest TER: {lowest_ter['metrics']['ter']:.2f} (step {lowest_ter['step']})")
        
        if best_bleu['metrics'].get('bertscore'):
            best_bert = max(self.metrics_history, 
                          key=lambda x: x['metrics'].get('bertscore', {}).get('f1', 0))
            print(f"Best BERTScore F1: {best_bert['metrics']['bertscore']['f1']:.2f} (step {best_bert['step']})")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    from pt_app.data.opus_dataset import LanguageDS
    
    # Initialize trainer with MT metrics enabled
    lora = LloraTrainer(
        enable_mt_metrics=True,
        source_lang="en",
        target_lang="pt"
    )
    
    m, t = lora.get_model()
    lora.get_optimizer()
    
    train, val = LanguageDS(tokenizer=t, dataset='opus_books').create_datasets(save=True)
    
    # Train with integrated MT metrics
    adapter_path = lora.train(train_set=train, val_set=val)