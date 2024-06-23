import torch 

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


class LayerAverage(torch.nn.Module):
    def __init__(self, key_in: str, key_out: str, audio_model_layers: int):
        super().__init__()
        self.key_in, self.key_out = key_in, key_out

        self.avg_weights = torch.nn.Parameter(
            torch.ones(
                len(audio_model_layers),
            )
        )

    def forward(self, x):
        """
        Layers weighted average
        """
        w = torch.nn.functional.softmax(self.avg_weights, dim=0)

        x[self.key_out] = torch.sum(
            x[self.key_in] * w[None, :, None, None],
            dim=1,
        )