# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from ..fondation.batch import collate


def generate_embeddings_dataset(dataset, encoder, batch_size, cuda, parallel):
    """
    Pre-compute all the embeddings for a given dataset.

    Arguments:
        dataset (Dataset): dataset to compute the embeddings for. It should
            contain a `'wav'` tensor
        encoder (sing.ae.models.ConvolutionalEncoder):
            encoder to use to generate the embedding
        batch_size (int): batch size to use
        cuda (bool): if `True`, performs the computation on GPU
        parallel (bool): if `True`, use all available GPUs

    Returns:
        Dataset: dataset of the same size as `dataset` but with the `'wav'`
            tensor replaced by an `'embeddings'` tensor.

    """

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    embeddings_dataset = [None] * len(dataset)

    if cuda:
        encoder.cuda()
    if parallel:
        encoder = nn.DataParallel(encoder)

    row = 0
    with tqdm.tqdm(total=len(dataset), unit="ex") as bar:
        for batch in loader:
            if cuda:
                batch.cuda_()
            with torch.no_grad():
                batch.tensors['embeddings'] = encoder.forward(
                    batch.tensors['wav'])
            del batch.tensors['wav']

            for item in batch.cpu():
                embeddings_dataset[row] = item
                row += 1
            bar.update(len(batch))
    return embeddings_dataset
