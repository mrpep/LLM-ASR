from pathlib import Path
from pandas import read_csv
from numpy import array
from torch import LongTensor


def librispeech_to_csv(data_path, split, audio_format='opus'):
    split_path = Path(data_path, split)
    split_transcripts = read_csv(split_path / "transcripts.txt", sep="\t", header=None)
    split_transcripts[2] = split_transcripts[0].apply(lambda x:
                                                      f"{split_path}/audio/{x[:-6].replace('_', '/')}{x}.{audio_format}")
    split_transcripts.columns = ['id', 'text', 'path']
    split_transcripts.to_csv(split_path / 'transcripts.csv', index=False)


def collate(batch, tokenizer):
    audios, texts, texts_lens, speech_lens = [], [], [], []
    for sample in batch:
        texts.append(sample['text'] + tokenizer.eos_token)
        texts_lens.append(len(sample['text']))
        audios.append(sample['audio'])
        speech_lens.append(sample['audio'].shape[1])
    inputs = tokenizer(texts, add_special_tokens=True, padding=True, truncation=False)
    # inputs['attention_mask'] es una máscara booleana que indica dónde se padeó
    # puede ser mejor que usar texts_lens en prepare_input de LLMASR
    return LongTensor(array(audios)), LongTensor(inputs['input_ids']), LongTensor(speech_lens), LongTensor(texts_lens)

