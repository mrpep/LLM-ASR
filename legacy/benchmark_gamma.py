from transformers import AutoModelForCausalLM
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader, Dataset
import torchinfo

class GenericModel(pl.LightningModule):
    def __init__(self, model,
                        model_dim=768,
                        lora_config = None, 
                        min_seqlen=100, 
                        step_seqlen=100, 
                        n_benchmark_per_step=10, 
                        batch_size=4):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model, attn_implementation="flash_attention_2")
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
        
        self.batch_size=batch_size
        self.seqlen = min_seqlen
        self.step_seqlen = step_seqlen
        self.n_benchmark_per_step = n_benchmark_per_step
        self.benchmark_step = 0
        self.model_dim = model_dim

    def forward(self, x):
        return self.model(inputs_embeds=x, attention_mask=torch.ones((x.shape[0],x.shape[1]), device=x.device))['logits']

    def training_step(self, batch, batch_idx):
        print(f'Batch ({self.batch_size},{self.seqlen},{self.model_dim})')
        x = torch.randn((self.batch_size, self.seqlen, self.model_dim), device=self.device, dtype=self.dtype)
        yhat = self(x)
        y = torch.randint(low=0, high=100, size=(self.batch_size, self.seqlen), device=self.device, dtype=torch.int64)
        loss = torch.nn.functional.cross_entropy(yhat.transpose(1,2), y)
        self.seqlen += self.step_seqlen
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

lora_config = LoraConfig(
                        r=16,
                        lora_alpha=16,
                        target_modules=["q_proj", "v_proj", "k_proj"],
                        lora_dropout=0.1
                    )
# #For Phi3-Mini:
# lora_config = LoraConfig(
#                         r=16,
#                         lora_alpha=16,
#                         target_modules=["qkv_proj"],
#                         lora_dropout=0.1
#                     )
# #For Phi3-Small:
# lora_config = LoraConfig(
#                         r=16,
#                         lora_alpha=16,
#                         target_modules=["query_key_value"],
#                         lora_dropout=0.1
#                     )
model = GenericModel('mistralai/Mistral-7B-v0.3', batch_size=4, model_dim=4096, lora_config=lora_config)

trainer = pl.Trainer(devices=1, accelerator='auto',precision='bf16-true')

class DummyDataset(Dataset):
    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return 100000

dl = DataLoader(DummyDataset(), batch_size=1)
trainer.fit(model, dl)