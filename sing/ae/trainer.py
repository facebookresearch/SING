# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from ..fondation import utils, trainer


class AutoencoderTrainer(trainer.BaseTrainer):
    """
    Trainer for the autoencoder.
    """

    def _train_batch(self, batch):
        rebuilt, target = self._get_rebuilt_target(batch)
        self.optimizer.zero_grad()
        loss = self.train_loss(rebuilt, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_rebuilt_target(self, batch):
        wav = batch.tensors['wav']
        target = utils.unpad1d(wav, self.model.decoder.strip)
        rebuilt = self.parallel.forward(wav)
        return rebuilt, target
