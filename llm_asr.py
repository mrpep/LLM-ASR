from abc import abstractmethod
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, WavLMModel
import pytorch_lightning as pl
import torch
import transformers


class HFLLMModel(torch.nn.Module):
    def __init__(self, hf_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(hf_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)

    def forward(self, x, attention_mask, **kwargs):
        return self.model(inputs_embeds=x, attention_mask=attention_mask, **kwargs)

    @abstractmethod
    def get_lut(self):
        pass

class WavLM(torch.nn.Module):
    def __init__(self, hf_path, layer=10):
        super().__init__()
        self.model = WavLMModel.from_pretrained(hf_path)
        self.downsampling = 320
        self.layer = layer

    def forward(self, x):
        return torch.stack(self.model(x, output_hidden_states=True)['hidden_states'])[self.layer]

class GPTModel(HFLLMModel):
    def get_lut(self):
        if 'Peft' in self.model.__class__.__name__:
            model_lut = self.model.model
        else:
            model_lut = self.model
        if model_lut.__class__.__name__ in ['LlamaForCausalLM','Qwen2ForCausalLM']:
            return model_lut.model.embed_tokens
        else:
            return model_lut.transformer.wte
    
    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

class LLMASR(pl.LightningModule):
    def __init__(self, llm_model, wav_model, lr, warmup_steps):
        super().__init__()
        self.llm_model = llm_model
        self.llm_model_lut = self.llm_model.get_lut()
        self.wav_model = wav_model
        self.wav_projector = torch.nn.Sequential(torch.nn.Linear(self.wav_model.model.config.hidden_size, 1024),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1024,self.llm_model.model.config.hidden_size))
        self.lr = lr
        self.warmup_steps = warmup_steps

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler_config = {'scheduler': transformers.get_linear_schedule_with_warmup(optimizer, self.warmup_steps, 100000),
                               'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def prepare_input(self, speech, transcription, speech_lens, transcription_lens, instructions):
        with torch.no_grad():
            x = []
            y = []
            
            speech = self.wav_model(speech)
            #Adapter muy sencillo para arrancar:
            speech = torch.nn.functional.avg_pool1d(speech.transpose(1,2), kernel_size=4, stride=4).transpose(1,2)
            speech = self.wav_projector(speech)
            speech_lens = speech_lens // (self.wav_model.downsampling*4) - 1
            transcription_embeds = self.llm_model_lut(transcription)
            instruction_embeds = self.llm_model_lut(instructions)
            for s, sl, t, te, tl, ins in zip(speech, speech_lens, transcription, transcription_embeds, transcription_lens, instruction_embeds):
                si = s[:sl]
                ti = te[:tl]
                xi = torch.cat([ins, si, ti], axis=0)
                yi = torch.cat([torch.ones((sl + ins.shape[0],), device=xi.device, dtype=torch.long)*-100, t[1:tl], torch.tensor([-100], device=xi.device, dtype=torch.long)])
                x.append(xi)
                y.append(yi)
            xlens = [len(xi) for xi in x]
            maxlen = max(xlens)
            x = [torch.nn.functional.pad(xi, (0, 0, 0, maxlen - xi.shape[0])) for xi in x]
            y = [torch.cat([yi,torch.tensor([-100]*(maxlen-yi.shape[0]),dtype=yi.dtype,device=yi.device)]) for yi in y]

            xlens = torch.tensor(xlens, device=speech.device)
            padding_mask = torch.arange(0, maxlen, device=speech.device)[None, :] < xlens[:, None]

        return torch.stack(x), torch.stack(y), padding_mask

    def forward(self, x):
        
        xin, y, padding_mask = self.prepare_input(x['speech'], 
                                                x['transcription'], 
                                                x['speech_lens'], 
                                                x['transcription_lens'],
                                                x['instruction'])
        
        return self.llm_model(xin[:, :-1], attention_mask=padding_mask[:,:-1]), y

    def generate(self, x):
        speech = self.wav_model(speech)
        speech = torch.nn.functional.avg_pool1d(speech.transpose(1,2), kernel_size=4, stride=4).transpose(1,2)
        speech = self.wav_projector(speech)
        self.llm_model.generate(speech, generation_config)

    def training_step(self, batch, batch_idx):
        ypred, ytrue = self(batch)
        yhat = ypred['logits']
        loss = torch.nn.functional.cross_entropy(yhat.transpose(1,2),ytrue[:,:-1])
        self.log('training_loss', loss)
        return loss
