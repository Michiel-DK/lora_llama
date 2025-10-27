from pt_app.trainer.trainer import LloraTrainer  # Import your existing trainer
from pt_app.metrics import MTEvaluator, MTTrainingCallback
from params import *
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm_lora.trainer.datasets import CacheDataset

from params import *
import math
from datetime import datetime

class LloraTrainerWithMT(LloraTrainer):
    """Extended trainer with MT metrics"""
    
    def __init__(self):
        super().__init__()
        # Initialize MT evaluator
        self.mt_evaluator = MTEvaluator(
            source_lang="en",
            target_lang="pt", 
            use_bert_score=True
        )
    
    def train(self, train_set, val_set, use_mt_metrics=True):
        """Train with optional MT metrics"""
        num_samples = len(train_set)
        batches_per_epoch = math.ceil(num_samples / BATCH_SIZE)
        iters = EPOCHS * batches_per_epoch
        print(f"[INFO] Calculated {iters} iterations from {EPOCHS} epochs (dataset size: {num_samples}, batch size: {self.batch_size})")
        
        timing = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_adapter_path = os.path.join(ADAPTER_PATH, f'{timing}_epoch{iters}.safetensors')
        
        # Choose callback based on use_mt_metrics flag
        if use_mt_metrics:
            callback = MTTrainingCallback(
                model=self.model,
                tokenizer=self.tokenizer,
                mt_evaluator=self.mt_evaluator,
                log_dir=os.path.join("./logs", f"mt_{timing}"),
                eval_samples=50,  # Adjust based on your needs
                prompt_template="Translate to Portuguese: {source}\n\nPortuguese:"
            )
            print("[INFO] Training with MT metrics enabled")
        else:
            callback = TrainingCallback()
            
    # After training, directly save metrics if available 
        try:
            if use_mt_metrics:
                # Debug print callback object
                print(f"[DEBUG] Callback after training: {callback}")
                print(f"[DEBUG] Has metrics_history: {'Yes' if hasattr(callback, 'metrics_history') else 'No'}")
                print(f"[DEBUG] metrics_history length: {len(callback.metrics_history) if hasattr(callback, 'metrics_history') else 'N/A'}")
                
                # Direct save debug file
                with open(os.path.join("./logs", f"mt_{timing}", "debug_after_training.txt"), 'w') as f:
                    f.write(f"Callback metrics_history: {callback.metrics_history}\n")
        except Exception as e:
            print(f"[DEBUG] Error accessing callback after training: {e}")
    
        
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
        
        # Train with selected callback
        train_sft(
            model=self.model,
            args=sft_args,
            optimizer=self.opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(val_set),
            training_callback=callback
        )
        
        # Print final MT metrics if available
        import ipdb;ipdb.set_trace()
        if use_mt_metrics and isinstance(callback, MTTrainingCallback):
                print("[DEBUG] Manually forcing metrics computation")
                # Manually invoke the metrics computation
                try:
                    callback.on_eval_end(
                        step=sft_args.iters,  # Final step
                        val_dataset=CacheDataset(val_set),
                        val_loss=0.0  # We don't know the loss here
                    )
                except Exception as e:
                    print(f"[DEBUG] Error in manual metrics computation: {e}")
                    import traceback
                    traceback.print_exc()
                
        return new_adapter_path

if __name__ == '__main__':
    from pt_app.data.opus_dataset import LanguageDS
    
    lora = LloraTrainerWithMT()
    
    m, t = lora.get_model()
    lora.get_optimizer()
    
    train, val = LanguageDS(tokenizer=t, dataset='opus_books').create_datasets(save=True)
    
    import ipdb;ipdb.set_trace()
    
    adapter_path = lora.train(train_set=train, val_set=val, use_mt_metrics=True)
