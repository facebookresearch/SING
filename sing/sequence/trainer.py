# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import nn

from ..fondation import utils, trainer


class SequenceGeneratorTrainer(trainer.BaseTrainer):
    """
    Trainer for the sequence generator (LSTM) part of SING.

    Arguments:
        decoder (sing.ae.models.ConvolutionalDecoder):
            decoder, used to compute the metrics on the waveforms
        truncated_gradient (int): size of sequence to compute
            the gradients over. If `None`, the whole sequence is used

    """

    def __init__(self, decoder, truncated_gradient=32, **kwargs):
        super(SequenceGeneratorTrainer, self).__init__(**kwargs)
        self.truncated_gradient = truncated_gradient
        self.decoder = decoder
        if self.is_parallel:
            self.parallel_decoder = nn.DataParallel(decoder)
        else:
            self.parallel_decoder = decoder

    def _train_batch(self, batch):
        embeddings = batch.tensors['embeddings']
        assert embeddings.size(-1) == self.model.length
        total_length = self.model.length
        hidden = None

        if self.truncated_gradient:
            truncated_gradient = self.truncated_gradient
        else:
            truncated_gradient = total_length

        steps = list(range(0, total_length, truncated_gradient))
        total_loss = 0
        for start_time in steps:
            sequence_length = min(truncated_gradient,
                                  total_length - start_time)
            target = embeddings[..., start_time:start_time + sequence_length]
            rebuilt, hidden = self.parallel.forward(
                start=start_time,
                length=sequence_length,
                hidden=hidden,
                **batch.tensors)
            hidden = tuple([h.detach() for h in hidden])
            self.optimizer.zero_grad()
            loss = self.train_loss(rebuilt, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() / len(steps)
        return total_loss

    def _get_rebuilt_target(self, batch):
        wav = batch.tensors['wav']
        target = utils.unpad1d(wav, self.decoder.strip)
        embeddings, _ = self.parallel.forward(**batch.tensors)
        rebuilt = self.parallel_decoder.forward(embeddings)
        return rebuilt, target


class SINGTrainer(trainer.BaseTrainer):
    """
    Trainer for the entire SING model.
    """

    def _get_rebuilt_target(self, batch):
        wav = batch.tensors['wav']
        rebuilt = self.parallel.forward(**batch.tensors)
        target = utils.unpad1d(wav, self.model.decoder.strip)
        return rebuilt, target
