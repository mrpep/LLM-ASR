from abc import abstractmethod
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, WavLMModel
import pytorch_lightning as pl
import torch


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
    def __init__(self, hf_path, layer=12):
        super().__init__()
        self.model = WavLMModel.from_pretrained(hf_path)
        self.downsampling = 320
        self.layer = layer

    def forward(self, x):
        return torch.stack(self.model(x, output_hidden_states=True)['hidden_states'])[self.layer]


class GPTModel(HFLLMModel):
    def get_lut(self):
        return self.model.transformer.wte


class LLMASR(pl.LightningModule):
    def __init__(self, llm_model, wav_model, lr):
        super().__init__()
        self.llm_model = llm_model
        self.llm_model_lut = self.llm_model.get_lut()
        self.wav_model = wav_model
        self.lr = lr

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_input(self, speech, transcription, speech_lens, transcription_lens):
        x = []
        speech = self.wav_model(speech)
        #Adapter muy sencillo para arrancar:
        speech = torch.nn.functional.avg_pool1d(speech.transpose(1,2), kernel_size=4, stride=4).transpose(1,2)
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
