from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from yaml import safe_load

def extract_audio_metadata(path):
    info = sf.info(path)
    return {'filename':path, 'samplerate': info.samplerate, 'duration': info.duration}

def read_mls(path):
    all_dfs = []
    for split in ['dev', 'train', 'test']:
        df = pd.read_csv(Path(path,split,'transcripts.txt'),delimiter='\t',header=None,names=['idx','transcription'])
        all_wavs = Path(path,split,'audio').rglob('*.opus')
        wav_mapping = {x.stem: str(x.resolve()) for x in all_wavs}
        df['filename'] = df['idx'].apply(lambda x: wav_mapping[x])
        df['partition'] = split
        df['start'] = 0
        all_dfs.append(df)
    df = pd.concat(all_dfs)
    metadatas = []
    for f in tqdm(df['filename']):
        metadatas.append(extract_audio_metadata(f))
    metadatas = pd.DataFrame(metadatas)
    return pd.merge(df, metadatas, left_on='filename', right_on='filename')

def read_tedlium(path):
    all_dfs = []
    for split in ['valid', 'train', 'test']:
        txt_path, wav_path = Path(path, split, 'txt'), Path(path, split, 'wav')
        transcripts = load_tedlium_transcripts(txt_path / f'{split}.es')
        with (txt_path / f'{split}.yaml').open('r') as f:
            audio_metadata = safe_load(f)
        for i in tqdm(range(len(audio_metadata))):
            audio = audio_metadata[i]
            audio['transcription'] = transcripts[i]
            audio['wav'] = audio['wav'].replace('wav', 'flac')
            audio['partition'] = split if split != 'valid' else 'dev'
            audio['filename'] = str((wav_path / audio['wav']).resolve())
            audio['samplerate'] = extract_audio_metadata(audio['filename'])['samplerate']
            audio['start'] = audio['offset']
            del audio['wav']
            del audio['offset']
            del audio['speaker_id']
        split_df = pd.DataFrame(audio_metadata).reset_index(names='idx')
        all_dfs.append(split_df)
    df = pd.concat(all_dfs)
    return df
        
                
def load_tedlium_transcripts(file_path):
    transcripts = []
    with file_path.open('r') as f:
        for line in f:
            line = line.lower()[:-2]
            transcripts.append(line)
    return transcripts