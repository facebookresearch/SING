# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from ..fondation import utils


class SequenceGenerator(nn.Module):
    """
    LSTM part of the SING model.

    Arguments:
        embeddings (dict[str, (int, int)]):
            represents the lookup tables used by the model.
            For each entry under the key `name`, with value
            `(cardinality, dimension)`, the tensor named `name` will be
            retrieved. Its values should be in `[0, cardinality - 1]`
            and the lookup table will have dimension `dimension`
        length (int): length of the generated sequence
        output_dimension (int): dimension of each generated sequence item
        hidden_size (int): size of each layer, see the documentation of
            :class:`nn.LSTM`
        num_layers (int): number of layers, see the documentation of
            :class:`nn.LSTM`

    """

    def __init__(self,
                 embeddings,
                 length,
                 time_dimension=4,
                 output_dimension=128,
                 hidden_size=1024,
                 num_layers=3):
        super(SequenceGenerator, self).__init__()
        self.tables = nn.ModuleList()
        self.inputs = []
        input_size = time_dimension
        for name, (cardinality, dimension) in sorted(embeddings.items()):
            input_size += dimension
            self.inputs.append(name)
            self.tables.append(
                nn.Embedding(
                    num_embeddings=cardinality, embedding_dim=dimension))

        if time_dimension == 0:
            self.time_table = None
        else:
            self.time_table = nn.Embedding(
                num_embeddings=length, embedding_dim=time_dimension)
        self.length = length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers)

        self.decoder = nn.Linear(hidden_size, output_dimension)

    def forward(self, start=0, length=None, hidden=None, **tensors):
        """
        Arguments:
            start (int): first time step to generate
            length (int): length of the sequence to generate. If `None`,
                will be taken to be `self.length - start`
            hidden ((torch.FloatTensor, torch.FloatTensor)):
                hidden state of the LSTM or `None` to start
                from a blank one
            **tensors (dict[str, torch.LongTensor]):
                dictionary containing the tensors used as inputs
                to the lookup tables specified by the `embeddings`
                parameter of the constructor
        """
        length = self.length - start if length is None else length

        inputs = []
        for name, table in zip(self.inputs, self.tables):
            value = tensors[name].transpose(0, 1)
            embedding = table.forward(value)
            inputs.append(embedding.expand(length, -1, -1))

        reference = inputs[0]
        if self.time_table is not None:
            times = torch.arange(
                start, start + length,
                device=reference.device).view(-1, 1).expand(
                    -1, reference.size(1))
            inputs.append(self.time_table.forward(times))
        input = torch.cat(inputs, dim=-1)
        if hidden is not None:
            hidden = [h.transpose(0, 1).contiguous() for h in hidden]

        self.lstm.flatten_parameters()
        output, hidden = self.lstm.forward(input, hidden)
        decoded = self.decoder(output.view(-1, output.size(-1))).view(
            output.size(0), output.size(1), -1)
        hidden = [h.transpose(0, 1) for h in hidden]
        return decoded.transpose(0, 1).transpose(1, 2), hidden


class SING(nn.Module):
    """
    Complete SING model.

    Arguments:
        sequence_generator (SequenceGenerator): the LSTM based
            sequence generator part of SING
        decoder (sing.ae.models.ConvolutionalDecoder):
            the convolutional decoder part of SING
    """

    def __init__(self, sequence_generator, decoder):
        super(SING, self).__init__()
        self.sequence_generator = sequence_generator
        self.decoder = decoder

    def forward(self, **tensors):
        """
        Arguments:
            **tensors (dict[str, torch.LongTensor]):
                Tensors used as inputs
                to the lookup tables specified by the `embeddings`
                parameter of :class:`SequenceGenerator`
        """
        return self.decoder.forward(self.sequence_generator(**tensors)[0])


def download_pretrained_model(target):
    """
    Download a pretrained version of SING.
    """
    url = "https://dl.fbaipublicfiles.com/sing/sing.th"
    sha256 = "eda8a7ce66f1ccf31cdd34a920290d80aabf96584c4d53df866b744f2862dc1c"
    utils.download_file(target, url, sha256=sha256)
