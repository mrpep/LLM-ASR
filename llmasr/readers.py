from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm

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
        all_dfs.append(df)
    df = pd.concat(all_dfs)
    metadatas = []
    for f in tqdm(df['filename']):
        metadatas.append(extract_audio_metadata(f))
    metadatas = pd.DataFrame(metadatas)
    return pd.merge(df, metadatas, left_on='filename', right_on='filename')