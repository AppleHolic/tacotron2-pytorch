import numpy as np
import librosa
import os

from typing import List, Tuple
from pytorch_sound.data.dataset import SpeechDataLoader
from pytorch_sound.data.meta.ljspeech import LJSpeechMeta
from torch.utils.data import Dataset
from pytorch_sound.utils.text import eng_t2i
from pytorch_sound.data.meta import MetaFrame


class SpeechTTSDataset(Dataset):

    def __init__(self, meta_frame: MetaFrame):
        self.meta_frame = meta_frame
        # column names
        self.audio_col = 'audio_filename'
        self.text_col = 'text'
        self.speaker_col = 'speaker'

    def __getitem__(self, idx: int) -> List:
        # indexing
        meta_item = self.meta_frame.iloc[idx]

        # parse meta information
        audio_file_path = meta_item[self.audio_col]
        raw_text = meta_item[self.text_col]

        # character indexing
        chr_indices = np.array(eng_t2i(raw_text))

        # load wave
        wav = self.load_audio(audio_file_path)

        # make masks
        wav_mask = np.ones_like(wav)
        chr_mask = np.ones_like(chr_indices)

        return wav, chr_indices, wav_mask, chr_mask

    def load_audio(self, file_path: str) -> np.ndarray:
        # Speed of librosa loading function is enhanced on version 0.7.0
        if file_path.endswith('.wav'):
            wav, sr = librosa.load(file_path, sr=None)
            assert sr == self.meta_frame.sr, \
                'sample rate miss match.\n {}\t {} in {}'.format(self.meta_frame.sr, sr, file_path)
        elif file_path.endswith('.npy'):
            wav = np.load(file_path)
        else:
            raise NotImplementedError('{} : File Type is not implemented to load audio data !'.format(file_path))
        return wav

    def __len__(self) -> int:
        return len(self.meta_frame)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LJSpeechMeta.frame_file_names[1:]

    # load meta file
    train_meta = LJSpeechMeta(os.path.join(meta_dir, train_file))
    valid_meta = LJSpeechMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechTTSDataset(train_meta)
    valid_dataset = SpeechTTSDataset(valid_meta)

    # create data loader
    train_loader = SpeechDataLoader(
        train_dataset, batch_size=batch_size, is_bucket=True, num_workers=num_workers, n_buckets=5
    )
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, is_bucket=False, num_workers=num_workers)

    return train_loader, valid_loader
