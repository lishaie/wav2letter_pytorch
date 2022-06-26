# import json
import math
from typing import Tuple, List

import librosa
import numpy as np
# import scipy.signal
# from scipy.io import wavfile
# import soundfile as sf
import torch
import torch.nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd


class WavDataset(Dataset):
    def __init__(self, manifest_filepath: str, sample_rate: int, labels: List[str]):
        """
        Create a dataset for ASR. Returns tensors representing raw audio. Audio conf and labels can be re-used from the
        model.
        This currently does not support "offset" and "duration" columns in the manifest, and assumes each audio file
        contains just the speech corresponding to the text in the record, and nothing more.
        Arguments:
            manifest_filepath (string): path to the manifest. Each line must be a json containing fields
                "audio_filepath" and "text".
            labels (list): list containing all valid labels in the text.
        """
        super().__init__()
        assert manifest_filepath.endswith('.csv')
        self.df = pd.read_csv(manifest_filepath, index_col=0)
        self.records = self.df.to_dict('records')

        self.size = len(self.df)
        self.sample_rate = sample_rate
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

    def __getitem__(self, index):
        sample = self.records[index]
        audio_path, transcript = sample['audio_filepath'], sample['text']
        target = list(filter(None, (self.labels_map.get(x) for x in transcript)))
        target = torch.IntTensor(target)

        wav, sample_rate = torchaudio.load(audio_path, channels_first=True)
        assert sample_rate == self.sample_rate, \
            f'expected sample_rate {self.sample_rate} but found {sample_rate} ({audio_path})'
        assert wav.shape[0] == 1, f'only single-channel audio files are supported ({audio_path}, {wav.shape})'

        return wav[0], target, audio_path, transcript

    def __len__(self):
        return self.size


def _collate_fn(batch: List[Tuple[torch.Tensor, torch.IntTensor, str, str]]):
    wav_tensors, targets, file_pathes, transcripts = zip(*batch)

    wav_lens = torch.IntTensor([len(x) for x in wav_tensors])
    target_lens = torch.IntTensor([len(x) for x in targets])
    wav_batch = pad_sequence(wav_tensors, batch_first=True)
    target_batch = pad_sequence(targets, batch_first=True)

    return wav_batch, wav_lens, target_batch, target_lens, file_pathes, transcripts


class WavDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class SpectrogramExtractor(torch.nn.Module):
    def __init__(self, audio_conf, mel_spec=64):
        super().__init__()
        window_size_samples = int(audio_conf.sample_rate * audio_conf.window_size)
        window_stride_samples = int(audio_conf.sample_rate * audio_conf.window_stride)
        self.n_fft = 2 ** math.ceil(math.log2(window_size_samples))
        filterbanks = torch.tensor(
            librosa.filters.mel(sr=audio_conf.sample_rate,
                                n_fft=self.n_fft,
                                n_mels=mel_spec, fmin=0, fmax=audio_conf.sample_rate / 2),
            dtype=torch.float
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(audio_conf.window, None)
        window_tensor = window_fn(window_size_samples, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.get_spect = torchaudio.transforms.MelSpectrogram(
            # spect params
            n_fft=self.n_fft,
            hop_length=window_stride_samples,
            win_length=window_size_samples,
            center=True,
            # window=self.window.to(dtype=torch.float),
            window_fn=window_fn,
            power=2.0,
            # Mel params
            sample_rate=audio_conf.sample_rate,
            n_mels=mel_spec, f_max=audio_conf.sample_rate / 2,
        )

    def _get_spect(self, audio: Tensor) -> Tensor:
        dithering = 1e-5
        preemph = 0.97
        x = audio  # FIXME REMOVE
        # x = audio + torch.randn(audio.shape, device=audio.device) * dithering  # dithering FIXME UNCOMMENT
        # x = torch.cat((x[0].unsqueeze(0), x[1:] - preemph * x[:-1]), dim=0)  # preemphasi
        x = self.get_spect(x)
        # x = torch.matmul(self.fb.to(x.dtype), x)  # apply filterbanks
        return x

    def forward(self, signal):
        epsilon = 1e-5
        log_zero_guard_value = 2 ** -24
        spect = self._get_spect(signal)
        spect = torch.log1p(spect + log_zero_guard_value)
        # normalize across time, per feature
        mean = spect.mean(axis=-1, keepdims=True)
        std = spect.std(dim=-1, keepdim=True)
        std += epsilon
        # print('spect.shape:', spect.shape)
        # print('mean.shape: ', mean.shape)
        spect = spect - mean
        spect = spect / std
        return spect.squeeze()
