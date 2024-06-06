from abc import abstractmethod
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, WavLMModel
import pytorch_lightning as pl
import torch
import transformers


class HFLLMModel(torch.nn.Module):
    def __init__(self, hf_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(hf_path, attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)

    def forward(self, x, attention_mask, **kwargs):
        return self.model(inputs_embeds=x, attention_mask=attention_mask, **kwargs)

    @abstractmethod
    def get_lut(self):
        pass

class WavLM(torch.nn.Module):
    def __init__(self, hf_path, layer=12):
        super().__init__()
        self.model = WavLMModel.from_pretrained(hf_path)
        self.downsampling = 320
        self.layer = layer

    def forward(self, x):
        return torch.stack(self.model(x, output_hidden_states=True)['hidden_states'])[self.layer]


class GPTModel(HFLLMModel):
    def get_lut(self):
        if 'Llama' in self.model.__class__.__name__:
            return self.model.model.embed_tokens
        else:
            return self.model.transformer.wte


class LLMASR(pl.LightningModule):
    def __init__(self, llm_model, wav_model, lr, warmup_steps):
        super().__init__()
        self.llm_model = llm_model
        self.llm_model_lut = self.llm_model.get_lut()
        self.wav_model = wav_model
        self.wav_projector = torch.nn.Linear(self.wav_model.model.config.hidden_size, self.llm_model.model.model.config.hidden_size)
        self.lr = lr
        self.warmup_steps = warmup_steps

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_config = {'scheduler': transformers.get_linear_schedule_with_warmup(optimizer, self.warmup_steps, 100000),
                               'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def prepare_input(self, speech, transcription, speech_lens, transcription_lens):
        x = []
        with torch.no_grad():
            speech = self.wav_model(speech)
        #Adapter muy sencillo para arrancar:
        speech = torch.nn.functional.avg_pool1d(speech.transpose(1,2), kernel_size=4, stride=4).transpose(1,2)
        speech = self.wav_projector(speech)
        speech_lens = speech_lens // (self.wav_model.downsampling*4) - 1
        transcription = self.llm_model_lut(transcription)
        for s, sl, t, tl in zip(speech, speech_lens, transcription, transcription_lens):
            si = s[:sl]
            ti = t[:tl]
            xi = torch.cat([si, ti], axis=0)
            x.append(xi)
        xlens = [len(xi) for xi in x]
        maxlen = max(xlens)
        x = [torch.nn.functional.pad(xi, (0, 0, 0, maxlen - xi.shape[0])) for xi in x]
        xlens = torch.tensor(xlens, device=speech.device)
        speech_lens = torch.tensor(speech_lens, device=speech.device)
        padding_mask = torch.arange(0, maxlen, device=speech.device)[None, :] < xlens[:, None]
        response_mask = torch.logical_and(torch.arange(0, maxlen)[None, :].to(speech.device) >= speech_lens[:, None],
                                          torch.arange(0, maxlen)[None, :].to(speech.device) < xlens[:, None])
        return torch.stack(x), padding_mask, response_mask

    def forward(self, x):
        
        xin, padding_mask, response_mask = self.prepare_input(x['speech'], 
                                                              x['transcription'], 
                                                              x['speech_lens'], 
                                                              x['transcription_lens'])
        
        return self.llm_model(xin[:, :-1], attention_mask=padding_mask[:,:-1]), response_mask[:, 1:], xin[:, 1:]

    def training_step(self, batch, batch_idx):
        ypred, response_mask, ytrue = self(batch)
        yhat = ypred['logits'][response_mask]
        yt = batch['transcription'][batch['text_padding_mask'].bool()]
        loss = torch.nn.functional.cross_entropy(yhat,yt)
        self.log('training_loss', loss)
        return loss
