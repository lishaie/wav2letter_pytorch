# import json
import math
from typing import Tuple, List

import torch
import torch.nn
from torch import Tensor, IntTensor, nn
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


def _collate_fn(batch: List[Tuple[Tensor, IntTensor, str, str]]):
    wav_tensors, targets, file_paths, transcripts = zip(*batch)

    wav_lens = IntTensor([len(x) for x in wav_tensors])
    target_lens = IntTensor([len(x) for x in targets])
    wav_batch = pad_sequence(wav_tensors, batch_first=True)
    target_batch = pad_sequence(targets, batch_first=True)

    return wav_batch, wav_lens, target_batch, target_lens, file_paths, transcripts


class WavDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class SpectrogramExtractor(nn.Module):
    def __init__(self, audio_conf, mel_spec=64):
        super().__init__()
        window_size_samples = int(audio_conf.sample_rate * audio_conf.window_size)
        window_stride_samples = int(audio_conf.sample_rate * audio_conf.window_stride)
        self.n_fft = 2 ** math.ceil(math.log2(window_size_samples))

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(audio_conf.window, None)
        self.get_spect = torchaudio.transforms.MelSpectrogram(
            # spect params
            n_fft=self.n_fft,
            hop_length=window_stride_samples,
            win_length=window_size_samples,
            center=True,
            window_fn=window_fn,
            power=2.0,
            # Mel params
            sample_rate=audio_conf.sample_rate,
            n_mels=mel_spec, f_max=audio_conf.sample_rate / 2,
        )

    def forward(self, signal):
        epsilon = 1e-5
        log_zero_guard_value = 2 ** -24
        spect = self.get_spect(signal)
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
