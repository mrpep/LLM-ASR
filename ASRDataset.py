import pandas as pd
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset


def librispeech_to_csv(data_path, split):
    split_path = Path(data_path, split)
    split_transcripts = pd.read_csv(split_path / "transcripts.txt", sep="\t", header=None)
    split_transcripts[2] = split_transcripts[0].apply(lambda x: f"{split_path}/audio/{x[:-6].replace('_', '/')}{x}.opus")
    split_transcripts.columns = ['id', 'text', 'path']
    split_transcripts.to_csv(split_path / 'transcripts.csv', index=False)
    

class ASRDataset(Dataset):
    def __init__(self, root_path, split, sr=16000):
        split_path = Path(root_path, split)
        self.data = pd.read_csv(split_path / 'transcripts.csv')
        self.sr = sr
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = torchaudio.load(row['path'])
        if sr != self.sr:
            audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
        return audio, row['text']