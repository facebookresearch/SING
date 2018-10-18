# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random

import torch

from .utils import random_seed_manager


class DatasetSubset:
    """
    Represents a subset of a dataset.

    Arguments:
        dataset (Dataset): dataset to take a subset of.
        indexes (list[int]): list of indexes to keep.
    """

    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = torch.LongTensor(indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]


class RandomSubset(DatasetSubset):
    """
    A random subset of a given size built from another dataset.

    Arguments:
        dataset (Dataset): dataset to take a random subset of.
        size (int): size of the random subset
        random_seed (int): random seed used to select the indexes.
    """

    def __init__(self, dataset, size, random_seed=42):
        indexes = list(range(len(dataset)))
        with random_seed_manager(random_seed):
            random.shuffle(indexes)

        super(RandomSubset, self).__init__(
            dataset=dataset, indexes=indexes[:size])
