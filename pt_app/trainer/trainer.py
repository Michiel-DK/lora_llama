
from mlx_lm_lora.utils import from_pretrained
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm_lora.trainer.sft_trainer import SFTTrainingArgs, train_sft
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm_lora.trainer.datasets import CacheDataset
import mlx.optimizers as optim


from params import *
import math
from datetime import datetime

class LloraTrainer():
    
    def __init__(self):
        
        self.model_name = MODEL_NAME
        self.lora_config = LORA_CONFIG
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
    def get_model(self):
    
        self.model, self.tokenizer = from_pretrained(
                model=self.model_name,
                lora_config=self.lora_config,
                quantized_load={
                    "bits": 4,
                    "group_size": 64
                },
            )
        print_trainable_parameters(self.model)
        
        return self.model, self.tokenizer
    
    def get_optimizer(self):
        self.opt = optim.AdamW(learning_rate=1e-5)

    
    def train(self, train_set, val_set):
        
        num_samples = len(train_set)
        batches_per_epoch = math.ceil(num_samples / BATCH_SIZE)
        iters = EPOCHS * batches_per_epoch
        print(f"[INFO] Calculated {iters} iterations from {EPOCHS} epochs (dataset size: {num_samples}, batch size: {self.batch_size})")
        
        timing = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_adapter_path = os.path.join(ADAPTER_PATH, f'{timing}_epoch{iters}.safetensors')
        
        train_sft(
            model=self.model,
            args=SFTTrainingArgs(
                batch_size=BATCH_SIZE,
                iters=iters,
                val_batches=1,
                steps_per_report=20,
                steps_per_eval=50,
                steps_per_save=50,
                adapter_file=new_adapter_path,
                max_seq_length=MAX_SEQ_LENGTH,
                grad_checkpoint=True,
            ),
            optimizer=self.opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(val_set),
            training_callback=TrainingCallback()
        )
        
if __name__ == '__main__':
    from pt_app.data.opus_dataset import OpusDS
    
    lora = LloraTrainer()
    
    m, t = lora.get_model()
    lora.get_optimizer()
    
    train, val = OpusDS(tokenizer=t).create_datasets()
    
    lora.train(train_set=train, val_set=val)