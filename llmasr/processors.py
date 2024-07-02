import random
import numpy as np
from typing import Dict, Any, List, Union
from loguru import logger
import librosa

class ReadAudioProcessor:
    """Processor to read audio files."""

    def __init__(self, key_in: str, key_out: str, max_length: Union[float, None] = None, mono: bool = True, target_sr = 16000) -> None:
        """Initialize ReadAudioProcessor.

        Args:
            key_in (str): Key for input audio.
            key_out (str): Key for output audio.
            max_length (Union[float, None], optional): Maximum length of audio in seconds. Defaults to None.
            mono (bool, optional): Whether to convert stereo audio to mono. Defaults to True.
        """
        super().__init__()
        self.key_in, self.key_out, self.max_length, self.mono, self.target_sr = key_in, key_out, max_length, mono, target_sr

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Read audio file and process it.

        Args:
            x (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Processed data.
        """
        try:
            if self.max_length is not None:
                audio_info = sf.info(x[self.key_in])
                desired_frames = int(self.max_length*audio_info.samplerate)
                total_frames = audio_info.frames
                if total_frames > desired_frames:
                    start = random.randint(0,total_frames - desired_frames)
                    duration = desired_frames
                else:
                    start = 0
                    duration = None
            else:
                start = 0
                duration = None
            if 'start' in x:
                start = x['start']
            if 'stop' in x:
                duration = x['stop'] - x['start']
            if 'duration' in x:
                duration = x['duration']
                
            x['start'] = start
            x['duration'] = duration
            wav, fs = librosa.core.load(x[self.key_in], offset=start, duration=duration, dtype=np.float32, sr=self.target_sr)
            if (wav.ndim == 2) and self.mono:
                wav = np.mean(wav,axis=-1)
        except Exception as e:
            logger.warning('Failed reading {}'.format(x[self.key_in]))
            wav = None
        x[self.key_out] = wav
        return x

class AddConstantValue:
    def __init__(self, key_out, value):
        self.key_out, self.value = key_out, value

    def __call__(self, x):
        x[self.key_out] = self.value
        return x

class PrependValue:
    def __init__(self, key_in, key_out, value):
        self.key_in, self.key_out, self.value = key_in, key_out, value

    def __call__(self, x):
        x[self.key_out] = self.value + x[self.key_in]
        return x

class FilterByValue:
    def __init__(self, column, value, mode):
        self.column, self.value, self.mode = column, value, mode

    def __call__(self, df):
        if self.mode == 'lt':
            return df.loc[df[self.column] < self.value]
        elif self.mode == 'gt':
            return df.loc[df[self.column] > self.value]
        elif self.mode == 'eq':
            return df.loc[df[self.column] == self.value]
        else:
            raise Exception(f'Unrecognized mode: {self.mode}. Available modes are: lt, gt and eq')