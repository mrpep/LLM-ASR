from pandas import read_csv
from torch.utils.data import Dataset
import torchaudio
import soundfile as sf
import copy
import numpy as np
import torch

class DictDataset(Dataset):
    def __init__(self, metadata, processors, out_cols):
        super().__init__()
        self._metadata = metadata
        self._processors = [p() for p in processors]
        self._out_cols = out_cols
    
    def __getitem__(self, idx):
        row = copy.deepcopy(self._metadata.iloc[idx])
        for p in self._processors:
            row = p(row)
        out = {k: row[k] for k in self._out_cols}
        return out

    def __len__(self):
        return len(self._metadata)

class ASRDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 split, 
                 sr=16000, 
                 instruction="Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate transcription of the given speech input \n\n### Input:\n", 
                 response='### Response:\n'):

        split_path = root_path / split
        self.data = read_csv(split_path / 'transcripts.csv')
        self.sr = sr
        self.instruction = instruction
        self.response = response
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio, sr = sf.read(row['path'])
        if sr != self.sr:
            audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
        return {'audio': audio, 
                'transcription': '{}{}'.format(self.response, row['text']),
                'instruction': self.instruction}

class InstructionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audios, texts, texts_lens, speech_lens, instructions = [], [], [], [], []
        for sample in batch:
            texts.append(sample['transcription'] + self.tokenizer.eos_token)
            audios.append(sample['wav'])
            instructions.append(sample['instruction'])
            speech_lens.append(sample['wav'].shape[0])
        inputs = self.tokenizer(texts, add_special_tokens=True, padding=True, truncation=False)
        instructions = self.tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)
        texts_lens = np.array(inputs['attention_mask']).sum(axis=1)
        max_audio_len = max(speech_lens)
        audios = np.stack([np.pad(xi, (0,max_audio_len - xi.shape[0])) for xi in audios], dtype=np.float32)

        return {'speech': torch.from_numpy(audios), 
                'transcription': torch.tensor(inputs['input_ids'], dtype=torch.long), 
                'speech_lens': torch.tensor(speech_lens, dtype=torch.long), 
                'transcription_lens': torch.tensor(texts_lens, dtype=torch.long),
                'text_padding_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'instruction': torch.tensor(instructions['input_ids'], dtype=torch.long)}