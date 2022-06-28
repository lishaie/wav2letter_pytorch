# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:59:45 2020

@author: Assaf Mushkin
"""
import random
import math
import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as ptl
from hydra.utils import instantiate
from torch import Tensor, IntTensor

torch_windows = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
    'none': None,
}


class SpectrogramExtractor(nn.Module):
    def __init__(self, audio_conf, mel_spec=64):
        super().__init__()
        window_fn = torch_windows.get(audio_conf.window, None)
        self.window_size = int(audio_conf.sample_rate * audio_conf.window_size)
        self.window_stride = int(audio_conf.sample_rate * audio_conf.window_stride)
        self.n_fft = 2 ** math.ceil(math.log2(self.window_size))

        self.mel_spect = torchaudio.transforms.MelSpectrogram(
            # spect params
            n_fft=self.n_fft, hop_length=self.window_stride, win_length=self.window_size, center=True,
            window_fn=window_fn, power=2.0,
            # Mel params
            n_mels=mel_spec, sample_rate=audio_conf.sample_rate, f_max=audio_conf.sample_rate / 2,
        )

    def forward(self, signal: Tensor, input_lengths: IntTensor = None):
        epsilon = 1e-5
        log_zero_guard_value = 2 ** -24
        spect = self.mel_spect(signal)
        spect = torch.log1p(spect + log_zero_guard_value)
        # normalize across time, per feature
        mean = spect.mean(axis=-1, keepdims=True)
        std = spect.std(dim=-1, keepdim=True)
        std += epsilon
        # print('spect.shape:', spect.shape)
        # print('mean.shape: ', mean.shape)
        spect = spect - mean
        spect = spect / std

        if input_lengths is not None:
            input_lengths = torch.ceil((input_lengths - self.window_size + self.window_stride) / self.window_stride)
            input_lengths = input_lengths.to(dtype=torch.int32)

        return spect, input_lengths


class ConvCTCASR(ptl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.audio_conf = cfg.audio_conf
        self.labels = cfg.labels
        self.ctc_decoder = instantiate(cfg.decoder)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.print_decoded_prob = cfg.get('print_decoded_prob', 0)
        self.example_input_array = self.create_example_input_array()

        sample_rate, window_stride, window_size, window, n_mels = \
            cfg.audio_conf['sample_rate'], cfg.audio_conf['window_stride'], cfg.audio_conf['window_size'], \
            cfg.audio_conf['window'], cfg['input_size']

        self.window_size_samples = int(sample_rate * window_size)
        self.window_stride_samples = int(sample_rate * window_stride)
        self.mel_spect = SpectrogramExtractor(cfg.audio_conf, mel_spec=n_mels)

    def create_example_input_array(self):
        batch_size = 4
        min_length, max_length = 100, 200
        lengths = torch.randint(min_length, max_length, (4,))
        return torch.rand(batch_size, self._cfg.input_size, max_length), lengths

    def compute_output_lengths(self, input_lengths: torch.IntTensor):
        """
        Compute the output lengths given the input lengths.
        Override if ratio is not strictly proportional (can happen with unpadded convolutions)
        """
        output_lengths = input_lengths.div(self.scaling_factor, rounding_mode='trunc')
        return output_lengths
    
    @property 
    def scaling_factor(self):
        """
        Returns a ratio between input lengths and output lengths.
        In convolutional models, depends on kernel size, padding, stride, and dilation.
        """
        raise NotImplementedError()
        
    def forward(self, inputs, input_lengths):
        raise NotImplementedError()
        # returns output, output_lengths
        
    def add_string_metrics(self, out, output_lengths, texts, prefix):
        decoded_texts = self.ctc_decoder.decode(out, output_lengths)
        if random.random() < self.print_decoded_prob:
            print(f'reference: {texts[0]}')
            print(f'decoded  : {decoded_texts[0]}')
        wer_sum, cer_sum, wer_denom_sum, cer_denom_sum = 0, 0, 0, 0
        for expected, predicted in zip(texts, decoded_texts):
            cer_value, cer_denom = self.ctc_decoder.cer_ratio(expected, predicted)
            wer_value, wer_denom = self.ctc_decoder.wer_ratio(expected, predicted)
            cer_sum += cer_value
            cer_denom_sum += cer_denom
            wer_sum += wer_value
            wer_denom_sum += wer_denom
        cer = cer_sum / cer_denom_sum
        wer = wer_sum / wer_denom_sum
        lengths_ratio = sum(map(len, decoded_texts)) / sum(map(len, texts))
        return {prefix+'_cer': cer, prefix+'_wer': wer, prefix+'_len_ratio': lengths_ratio}

    # PyTorch Lightning methods
    def configure_optimizers(self):
        optimizer = instantiate(self._cfg.optimizer, params=self.parameters())
        scheduler = instantiate(self._cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        with torch.no_grad():
            inputs, input_lengths = self.mel_spect(inputs, input_lengths)

        out, output_lengths = self.forward(inputs, input_lengths)
        loss = self.criterion(out.transpose(0, 1), targets, output_lengths, target_lengths)
        logs = {'train_loss': loss, 'learning_rate': self.optimizers().param_groups[0]['lr']}
        # logs.update(self.add_string_metrics(out, output_lengths, texts, 'train'))
        self.log_dict(logs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, file_paths, texts = batch
        with torch.no_grad():
            inputs, input_lengths = self.mel_spect(inputs, input_lengths)

        out, output_lengths = self.forward(inputs, input_lengths)
        loss = self.criterion(out.transpose(0, 1), targets, output_lengths, target_lengths)
        logs = {'val_loss': loss}
        logs.update(self.add_string_metrics(out, output_lengths, texts, 'val'))
        self.log_dict(logs)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
