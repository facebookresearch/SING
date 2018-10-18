# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import tqdm

from . import utils
from .batch import collate


class BaseTrainer:
    """
    Base class for all the epoch-based trainers. Takes care of various
    task common to all training like checkpointing,
    iterating over the different datasets, computing evaluation metrics etc.

    Arguments:
        model (nn.Module): model to train
        train_loss (nn.Module): loss used for training
        eval_losses (dict[str, nn.Module]): dictionary of evaluation losses
        train_dataset (Dataset): dataset used for training
        eval_datasets (dict[str, Dataset]): dictionary of datasets
            on which each evaluation loss will be computed
        epochs (int): number of epochs to train for
        suffix (str): suffix used for logging, for instance
            if the suffix is `"_phase1"`, during training, `"train_phase1"`
            will be displayed
        batch_size (int): batch size
        cuda (bool): if true, runs on GPU
        parallel (bool): if true, use all available GPUs
        lr (float): learning rate for :class:`optim.Adam`
        checkpoint_path (Path): path to save checkpoint to. If `None`, no
            checkpointing is performed. Otherwise, a checkpoint is saved
            at the end of each epoch and overwrites the previous one.

    """

    def __init__(self,
                 model,
                 train_loss,
                 eval_losses,
                 train_dataset,
                 eval_datasets,
                 epochs,
                 suffix="",
                 batch_size=32,
                 cuda=True,
                 parallel=False,
                 lr=0.0001,
                 checkpoint_path=None):
        self.model = model
        self.parallel = nn.DataParallel(model) if parallel else model
        self.is_parallel = parallel
        self.train_loss = train_loss
        self.eval_losses = nn.ModuleDict(eval_losses)
        self.batch_size = batch_size
        self.cuda = cuda
        self.suffix = suffix

        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=lr)

        if self.cuda:
            self.model.cuda()
            self.train_loss.cuda()
            self.eval_losses.cuda()
        else:
            self.model.cpu()
            self.train_loss.cpu()
            self.eval_losses.cpu()

    def _train_epoch(self, dataset, epoch):
        """
        Train a single epoch on the given dataset and displays
        statistics from time to time.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate)
        iterator = utils.progress_iterator(loader, divisions=20)

        total_loss = 0
        with tqdm.tqdm(total=len(dataset), unit="ex") as bar:
            for idx, (progress, batch) in enumerate(iterator):
                if self.cuda:
                    batch.cuda_()
                total_loss += self._train_batch(batch)
                bar.update(len(batch))
                if progress:
                    tqdm.tqdm.write(
                        "[train{}][{:03d}] {:.1f}%, loss {:.6f}".format(
                            self.suffix, epoch, progress,
                            total_loss / (idx + 1)))
        return total_loss

    def _eval_dataset(self, dataset_name, dataset, epoch):
        """
        Evaluate all the losses `eval_lossers` on the given dataset
        and reports the metrics averaged over the entire dataset.
        """
        loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collate)
        total_losses = {loss_name: 0 for loss_name in self.eval_losses}
        with tqdm.tqdm(total=len(dataset), unit="ex") as bar:
            for batch in loader:
                if self.cuda:
                    batch.cuda_()
                rebuilt, target = self._get_rebuilt_target(batch)
                for name, loss in self.eval_losses.items():
                    total_losses[name] += loss(rebuilt,
                                               target).item() * len(batch)
                bar.update(len(batch))

        print("[{}{}][{:03d}] Evaluation: \n{}\n".format(
            dataset_name, self.suffix, epoch, "\n".join(
                "\t{}={:.6f}".format(name, loss / len(dataset))
                for name, loss in total_losses.items())))
        return total_losses

    def _train_batch(self, batch):
        """
        Given a batch, call :meth:`_get_rebuilt_target`
        to obtain the `target` and `rebuilt` tensors and call
        :attr:`train_loss` on them, compute the gradient and perform
        one optimizer step.

        This method can be overriden in subclasses.
        """
        rebuilt, target = self._get_rebuilt_target(batch)
        self.optimizer.zero_grad()
        loss = self.train_loss(rebuilt, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_rebuilt_target(self, batch):
        """
        Should be implemenented in subclasses.
        Given a batch, returns a tuple (rebuilt, target).
        This tuple will be passed to all the losses in `eval_losses`.
        """
        raise NotImplementedError()

    def train(self):
        """
        Train :attr:`model` for :attr:`epochs`
        """
        last_epoch, state = utils.load_checkpoint(self.checkpoint_path)
        if state is not None:
            self.model.load_state_dict(state, strict=False)
        start_epoch = last_epoch + 1
        if start_epoch > self.epochs:
            raise ValueError(("Checkpoint has been trained for {} "
                              "epochs but we aim for {} epochs").format(
                                  start_epoch, self.epochs))
        if start_epoch > 0:
            print("Resuming training at epoch {}".format(start_epoch))
        for epoch in range(start_epoch, self.epochs):
            self._train_epoch(self.train_dataset, epoch)
            utils.save_checkpoint(self.checkpoint_path, epoch,
                                  self.model.state_dict())
            with torch.no_grad():
                for name, dataset in self.eval_datasets.items():
                    self._eval_dataset(name, dataset, epoch)
