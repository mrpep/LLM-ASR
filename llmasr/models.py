import torch
from peft import get_peft_model, LoraConfig
import pytorch_lightning as pl    
from abc import abstractmethod
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, WavLMModel
from transformers import get_linear_schedule_with_warmup as get_linear_schedule_with_warmup_
from transformers import GenerationConfig
import gin

def get_linear_schedule_with_warmup(*arg, **kwargs):
    return get_linear_schedule_with_warmup_(*arg, **kwargs)

class HFLLMModel(torch.nn.Module):
    def __init__(self, hf_path, lora_config=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(hf_path)
        if lora_config is not None:
            lora_conf = lora_config()
            self.lora_config = LoraConfig(**lora_conf)
            if lora_conf['r'] > 0:
                self.model = get_peft_model(self.model, self.lora_config)

    def forward(self, x, attention_mask, **kwargs):
        return self.model(inputs_embeds=x, attention_mask=attention_mask, **kwargs)

    def get_lut(self):
        lut = self.model
        if self.model.__class__.__name__ == 'PeftModelForCausalLM':
            lut = self.model.model
        if self.model.__class__.__name__ == 'Qwen2ForCausalLM':
            lut = model.model.embed_tokens
        return lut

class LLMASR(pl.LightningModule):
    def __init__(self, llm, audio_encoder, adapters, optimizer, lr_scheduler=None):
        super().__init__()
        self.llm_model = llm()
        self.llm_model_lut = self.llm_model.get_lut()
        self.audio_encoder = audio_encoder()
        self.adapters = torch.nn.ModuleList([a() for a in adapters])
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_input(self, x):
        with torch.no_grad():
            X = []
            Y = []
            x['audio_features'] = self.audio_encoder(x['speech'])
            x['audio_feature_lens'] = x['speech_lens']//self.audio_encoder.downsampling - 1
            
            print("-------------------")
            print(x)
            for a in self.adapters:
                a(x)
            print(x)
            print("-------------------")
            x['transcription_embeds'] = self.llm_model_lut(x['transcription'])
            x['instruction_embeds'] = self.llm_model_lut(x['instruction'])

            for s, sl, t, te, tl, ins in zip(x['audio_features'], x['audio_feature_lens'], x['transcription'], x['transcription_embeds'], x['transcription_lens'], x['instruction_embeds']):
                si = s[:sl]
                ti = te[:tl]
                xi = torch.cat([ins, si, ti], axis=0)
                yi = torch.cat([torch.ones((sl + ins.shape[0],), device=xi.device, dtype=torch.long)*-100, t[1:tl], torch.tensor([-100], device=xi.device, dtype=torch.long)])
                X.append(xi)
                Y.append(yi)
            xlens = [len(xi) for xi in X]
            maxlen = max(xlens)
            X = [torch.nn.functional.pad(xi, (0, 0, 0, maxlen - xi.shape[0])) for xi in X]
            Y = [torch.cat([yi,torch.tensor([-100]*(maxlen-yi.shape[0]),dtype=yi.dtype,device=yi.device)]) for yi in Y]

            xlens = torch.tensor(xlens, device=x['speech'].device)
            padding_mask = torch.arange(0, maxlen, device=x['speech'].device)[None, :] < xlens[:, None]

        x['llm_in'] = torch.stack(X)
        x['llm_target'] = torch.stack(Y)
        x['llm_padding_mask'] = padding_mask

    def forward(self, x):
        self.prepare_input(x)
        x['llm_logits'] = self.llm_model(x['llm_in'][:, :-1], attention_mask=x['llm_padding_mask'][:,:-1])['logits']

    def training_step(self, batch, batch_idx):
        self(batch)
        loss = torch.nn.functional.cross_entropy(batch['llm_logits'].transpose(1,2),batch['llm_target'][:,:-1])
        self.log('training_loss', loss)
        return loss

    def generate(self, x, tokenizer=None, gen_config=None):
        if gen_config is None:
            gen_config = GenerationConfig(max_new_tokens=128, do_sample=True, temperature=0.1, min_new_tokens=5, num_beams=4, eos_token_id=tokenizer.eos_token_id)
        self.prepare_input(x)
        outs = self.llm_model.model.model.generate(inputs_embeds=x['llm_in'][:,:-1], generation_config = gen_config, tokenizer=tokenizer)
        return tokenizer.decode(outs[0])
    
    def validation_step(self, batch, batch_idx):
        self(batch)
        loss = torch.nn.functional.cross_entropy(batch['llm_logits'].transpose(1,2),batch['llm_target'][:,:-1])
        self.log('validation_loss', loss)

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())
        if self.lr_scheduler is not None:
            if self.lr_scheduler.__name__ == 'SequentialLR':
                binds = gin.get_bindings('torch.optim.lr_scheduler.SequentialLR')
                lr_scheduler = self.lr_scheduler(opt, schedulers=[s(opt) for s in binds['schedulers']])
            else:
                lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}
        return opt_config

class WavLM(torch.nn.Module):
    def __init__(self, hf_path):
        super().__init__()
        self.model = WavLMModel.from_pretrained(hf_path)
        self.downsampling = 320

    def forward(self, x):
        with torch.no_grad():
            return torch.stack(self.model(x, output_hidden_states=True)['hidden_states'])

class InputSelector(torch.nn.Module):
    def __init__(self, layer, key_in, key_out):
        super().__init__()
        self.layer, self.key_in, self.key_out = layer, key_in, key_out

    def forward(self, x):
        x[self.key_out] = x[self.key_in][self.layer]

class CatDownsample(torch.nn.Module):
    def __init__(self, rate, key_in, key_out, hiddens=None, dim_in=None):
        super().__init__()
        self.key_in, self.key_out, self.rate = key_in, key_out, rate
        if hiddens is not None:
            ch = [dim_in*self.rate] + hiddens
            self.projector = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(ci,co), torch.nn.ReLU()) for ci,co in zip(ch[:-1], ch[1:])])
        else:
            self.projector = None


    def forward(self, x):
        pad = self.rate - (x[self.key_in].shape[1] % self.rate)
        xin = torch.nn.functional.pad(x[self.key_in],(0,0,0,pad))
        xin = xin.reshape(xin.shape[0],xin.shape[1]//self.rate,-1)
        if self.projector is not None:
            xin = self.projector(xin)
        x[self.key_out] = xin
        x['audio_feature_lens'] = x['audio_feature_lens']//self.rate + (x['audio_feature_lens']%self.rate > 0).long()
