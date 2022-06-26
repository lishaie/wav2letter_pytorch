# -*- coding: utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, IntTensor
import numpy as np

from base_asr_models import ConvCTCASR
from data import wav_data_loader


class Conv1dBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, drop_out_prob=-1.0, dilation=1, bn=True,
                 activation_use=True):
        super(Conv1dBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.drop_out_prob = drop_out_prob
        self.dilation = dilation
        self.activation_use = activation_use
        self.padding = kernel_size[0]
        '''Padding Calculation'''
        input_rows = input_channels
        filter_rows = kernel_size[0]
        out_rows = (input_rows + stride - 1) // stride
        self.padding_rows = max(0, (out_rows - 1) * stride + (filter_rows - 1) * dilation + 1 - input_rows)
        if self.padding_rows > 0:
            if self.padding_rows % 2 == 0:
                self.paddingAdded = nn.ReflectionPad1d(self.padding_rows // 2)
            else:
                self.paddingAdded = nn.ReflectionPad1d((self.padding_rows // 2, (self.padding_rows + 1) // 2))
        else:
            self.paddingAdded = nn.Identity()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                               stride=stride, padding=0, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(num_features=output_channels, momentum=0.9, eps=0.001) if bn else nn.Identity()
        self.drop_out = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else nn.Identity()

    def forward(self, xs):
        xs = self.paddingAdded(xs)
        output = self.conv1(xs)
        output = self.batch_norm(output)
        output = self.drop_out(output)
        if self.activation_use:
            output = torch.clamp(input=output, min=0, max=20)
        return output


class Wav2Letter(ConvCTCASR):
    def __init__(self, cfg):
        super(Wav2Letter, self).__init__(cfg)
        self.mid_layers = cfg.mid_layers
        if not cfg.input_size:
            nfft = (self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
            self.input_size = int(1 + (nfft / 2))
        else:
            self.input_size = cfg.input_size

        layers = cfg.layers[: self.mid_layers]
        layer_size = self.input_size
        conv_blocks = []
        for idx in range(len(layers)):
            layer_params = layers[idx]  # TODO: can we use **layer_params here?
            layer = Conv1dBlock(input_channels=layer_size, output_channels=layer_params.output_size,
                                kernel_size=(layer_params.kernel_size,), stride=layer_params.stride,
                                dilation=layer_params.dilation, drop_out_prob=layer_params.dropout)
            layer_size = layer_params.output_size
            conv_blocks.append(('conv1d_{}'.format(idx), layer))
        last_layer = Conv1dBlock(input_channels=layer_size, output_channels=len(self.labels), kernel_size=(1,),
                                 stride=1, bn=False, activation_use=False)
        conv_blocks.append(('conv1d_{}'.format(len(layers)), last_layer))
        self.conv1ds = nn.Sequential(OrderedDict(conv_blocks))

    @property
    def scaling_factor(self):
        if not hasattr(self, '_scaling_factor'):
            strides = []
            for module in self.conv1ds.children():
                strides.append(module.conv1.stride[0])
            self._scaling_factor = int(np.prod(strides))
        return self._scaling_factor

    def forward(self, x, input_lengths=None):
        x = self.conv1ds(x)
        x = x.transpose(1, 2)
        x = F.log_softmax(x, dim=-1)
        if input_lengths is not None:
            output_lengths = self.compute_output_lengths(input_lengths)
        else:
            output_lengths = None
        return x, output_lengths


class RawWav2Letter(Wav2Letter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.example_input_array = (torch.rand(4, 48000), torch.randint(16000, 48000, [4]))

        sample_rate, window_stride, window_size, window, n_mels = \
            cfg.audio_conf['sample_rate'], cfg.audio_conf['window_stride'], cfg.audio_conf['window_size'], \
            cfg.audio_conf['window'], cfg['input_size']

        self.window_size_samples = int(sample_rate * window_size)
        self.window_stride_samples = int(sample_rate * window_stride)
        # n_fft = 2 ** ceil(log2(self.window_size_samples))
        #
        # window_fn = {
        #     'hann': torch.hann_window,
        #     'hamming': torch.hamming_window,
        #     'blackman': torch.blackman_window,
        #     'bartlett': torch.bartlett_window,
        #     'none': None,
        # }.get(window)
        #
        # self.mel_spect = MelSpectrogram(
        #     n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, win_length=self.window_size_samples,
        #     hop_length=self.window_stride_samples, window_fn=window_fn, f_max=sample_rate / 2)
        self.mel_spect = wav_data_loader.SpectrogramExtractor(cfg.audio_conf, mel_spec=n_mels)

    def forward(self, x: Tensor, input_lengths: IntTensor = None):
        window_size, window_stride = self.window_size_samples, self.window_stride_samples
        # print('RawWav2Letter: forward: x.shape:', x.shape)
        x = self.mel_spect(x)

        if input_lengths is not None:
            input_lengths = torch.ceil((input_lengths - window_size + window_stride) / window_stride)
            input_lengths = input_lengths.to(dtype=torch.int32)

        return super().forward(x, input_lengths)
