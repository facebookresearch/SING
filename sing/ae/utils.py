# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn
from torch.nn import functional as F

from .. import dsp


class WindowedConv1d(nn.Module):
    """
    Smooth a convolution using a window.

    Arguments:
            conv (nn.Conv1d): convolution module to wrap
            window_name (str or None): name of the window used to smooth
            the convolutions. See :func:`sing.dsp.get_window`
            squared (bool): if `True`, square the smoothing window
    """

    def __init__(self, conv, window_name='hann', squared=True):
        super(WindowedConv1d, self).__init__()
        self.window_name = window_name
        if squared:
            self.window_name += "**2"
        self.register_buffer('window',
                             dsp.get_window(
                                 window_name,
                                 conv.weight.size(-1),
                                 squared=squared))
        self.conv = conv

    def forward(self, input):
        weight = self.window * self.conv.weight
        return F.conv1d(
            input,
            weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            padding=self.conv.padding)

    def __repr__(self):
        return "WindowedConv1d(window={},conv={})".format(
            self.window_name, self.conv)


class WindowedConvTranpose1d(nn.Module):
    """
    Smooth a transposed convolution using a window.

    Arguments:
            conv (nn.Conv1d): convolution module to wrap
            window_name (str or None): name of the window used to smooth
            the convolutions. See :func:`sing.dsp.get_window`
            squared (bool): if `True`, square the smoothing window
    """

    def __init__(self, conv_tr, window_name='hann', squared=True):
        super(WindowedConvTranpose1d, self).__init__()
        self.window_name = window_name
        if squared:
            self.window_name += "**2"
        self.register_buffer('window',
                             dsp.get_window(
                                 window_name,
                                 conv_tr.weight.size(-1),
                                 squared=squared))
        self.conv_tr = conv_tr

    def forward(self, input):
        weight = self.window * self.conv_tr.weight
        return F.conv_transpose1d(
            input,
            weight,
            bias=self.conv_tr.bias,
            stride=self.conv_tr.stride,
            padding=self.conv_tr.padding,
            output_padding=self.conv_tr.output_padding,
            groups=self.conv_tr.groups,
            dilation=self.conv_tr.dilation)

    def __repr__(self):
        return "WindowedConvTranpose1d(window={},conv_tr={})".format(
            self.window_name, self.conv_tr)
