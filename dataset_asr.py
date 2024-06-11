from pandas import read_csv
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf


class ASRDataset(Dataset):
    def __init__(self, root_path, split, sr=16000):
        split_path = root_path / split
        self.data = read_csv(split_path / 'transcripts.csv')
        self.sr = sr
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['path'])
        if sr != self.sr:
            audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
        return {'audio': audio, 
                'transcription': '### Response:\n{}'.format(row['text']),
                'instruction': "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate transcription of the given speech input \n\n### Input:\n"}
