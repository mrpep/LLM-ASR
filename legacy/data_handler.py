from pathlib import Path
from pandas import read_csv
import numpy as np
import torch
from torch import LongTensor


def librispeech_to_csv(data_path, split, audio_format='opus'):
    split_path = Path(data_path, split)
    split_transcripts = read_csv(split_path / "transcripts.txt", sep="\t", header=None)
    split_transcripts[2] = split_transcripts[0].apply(lambda x:
                                                      f"{split_path}/audio/{x[:-6].replace('_', '/')}{x}.{audio_format}")
    split_transcripts.columns = ['id', 'text', 'path']
    split_transcripts.to_csv(split_path / 'transcripts.csv', index=False)


def collate(batch, tokenizer):
    audios, texts, texts_lens, speech_lens, instructions = [], [], [], [], []
    for sample in batch:
        texts.append(sample['transcription'] + tokenizer.eos_token)
        #texts_lens.append(len(sample['transcription']))
        audios.append(sample['audio'])
        instructions.append(sample['instruction'])
        speech_lens.append(sample['audio'].shape[0])
    inputs = tokenizer(texts, add_special_tokens=True, padding=True, truncation=False)
    instructions = tokenizer(instructions, add_special_tokens=True, padding=True, truncation=False)
    # inputs['attention_mask'] es una máscara booleana que indica dónde se padeó
    # puede ser mejor que usar texts_lens en prepare_input de LLMASR
    texts_lens = np.array(inputs['attention_mask']).sum(axis=1)
    max_audio_len = max(speech_lens)
    audios = np.stack([np.pad(xi, (0,max_audio_len - xi.shape[0])) for xi in audios], dtype=np.float32)

    return {'speech': torch.from_numpy(audios), 
            'transcription': LongTensor(inputs['input_ids']), 
            'speech_lens': LongTensor(speech_lens), 
            'transcription_lens': LongTensor(texts_lens),
            'text_padding_mask': LongTensor(inputs['attention_mask']),
            'instruction': LongTensor(instructions['input_ids'])}