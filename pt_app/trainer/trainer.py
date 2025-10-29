
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
        self.quantization_config = QUANTIZATION_CONFIG
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.max_seq_length = MAX_SEQ_LENGTH
        self.optimizer_config = OPTIMIZER_CONFIG
        self.training_args = TRAINING_ARGS
        self.adapter_path = ADAPTER_PATH

        # self.load_adapter = load_adapter
        # self.adapter_to_load = adapter_to_load
        
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
        
        # if self.load_adapter and self.adapter_to_load:
        #     print(f"[INFO] Loading pretrained adapter from: {self.adapter_to_load}")
        #     self.model.load_adapter(self.adapter_to_load)
        
        print_trainable_parameters(self.model)
        
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

    
    def train(self, train_set, val_set):
        
        num_samples = len(train_set)
        batches_per_epoch = math.ceil(num_samples / BATCH_SIZE)
        iters = EPOCHS * batches_per_epoch
        print(f"[INFO] Calculated {iters} iterations from {EPOCHS} epochs (dataset size: {num_samples}, batch size: {self.batch_size})")
        
        timing = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_adapter_path = os.path.join(ADAPTER_PATH, f'{timing}_epoch{iters}.safetensors')

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
            # Additional args if supported by your version:
            # warmup_steps=self.training_args.get("warmup_steps", 100),
            # grad_clip=self.training_args.get("grad_clip", 1.0),
        )
        
        train_sft(
            model=self.model,
            args=sft_args,
            optimizer=self.opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(val_set),
            training_callback=TrainingCallback()
        )
        
        return new_adapter_path
        
if __name__ == '__main__':
    from pt_app.data.opus_dataset import LanguageDS
    
    lora = LloraTrainer()
    
    m, t = lora.get_model()
    lora.get_optimizer()
    
    train, val = LanguageDS(tokenizer=t, dataset='opus_books').create_datasets(save=True)
    
    import ipdb;ipdb.set_trace()
    
    lora.train(train_set=train, val_set=val)