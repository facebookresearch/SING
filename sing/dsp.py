# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.nn.functional as F


def power(spec):
    """
    Given a complex spectrogram, return the power spectrum.

    Shape:
        - `spec`: `(*, 2, F, T)`
        - Output: `(*, F, T)`
    """
    return spec[..., 0]**2 + spec[..., 1]**2


def get_window(name, window_length, squared=False):
    """
    Returns a windowing function.

    Arguments:
        window (str): name of the window, currently only 'hann' is available
        window_length (int): length of the window
        squared (bool): if true, square the window

    Returns:
        torch.FloatTensor: window of size `window_length`
    """
    if name == "hann":
        window = torch.hann_window(window_length)
    else:
        raise ValueError("Invalid window name {}".format(name))
    if squared:
        window *= window
    return window


class STFT(nn.Module):
    """
    Compute the STFT.
    See :mod:`torch.stft` for a definition of the parameters.

    Arguments:
        n_fft (int):  performs a FFT over `n_fft` samples
        hop_length (int or None): stride of the STFT transform. If `None`
            uses `n_fft // 4`
        window_name (str or None): name of the window used for the STFT.
            No window is used if `None`.

    """

    def __init__(self, n_fft=1024, hop_length=None, window_name='hann'):
        super(STFT, self).__init__()
        assert n_fft % 2 == 0
        window = None
        if window_name is not None:
            window = get_window(window_name, n_fft)
        self.register_buffer("window", window)
        self.hop_length = hop_length or n_fft // 4
        self.n_fft = n_fft

    def forward(self, input):
        return torch.stft(
            input,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False)


class SpectralLoss(nn.Module):
    """
    Compute a loss between two log power-spectrograms.

    Arguments:
        base_loss (function): loss used to compare the log power-spectrograms.
            For instance :func:`F.mse_loss`
        epsilon (float): offset for the log, i.e. `log(epsilon + ...)`
        **kwargs (dict): see :class:`STFT`
    """

    def __init__(self, base_loss=F.mse_loss, epsilon=1, **kwargs):
        super(SpectralLoss, self).__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.stft = STFT(**kwargs)

    def _log_spectrogram(self, signal):
        return torch.log(self.epsilon + power(self.stft.forward(signal)))

    def forward(self, a, b):
        spec_a = self._log_spectrogram(a)
        spec_b = self._log_spectrogram(b)
        return self.base_loss(spec_a, spec_b)


def float_wav_to_short(wav):
    """
    Given a float waveform, return a short waveform.
    The input waveform will be clamped between -1 and 1.
    """
    return (wav.clamp(-1, 1) * (2**15 - 1)).short()
