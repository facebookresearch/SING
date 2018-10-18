# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data.dataloader import default_collate


class BatchItem:
    """
    Reprensents a single batch item. A :class:`Batch` can be built
    from multiple :class:`BatchItem` using :func:`collate`.

    Attributes:
        metadata (dict[str, object]): Contains all the metadata
            about the batch item. Those elements will not be
            collated together
            when building a batch
        tensors (dict[str, tensor]): Contains all the tensors
            for the batch item. Those elements will be collated
            together when building a batch using :func:`default_collate`.

    """

    def __init__(self, metadata=None, tensors=None):
        self.metadata = dict(metadata) if metadata else {}
        self.tensors = dict(tensors) if tensors else {}


def collate(items):
    """
    Collate together all the items into a :class:`Batch`.
    The metadata dictionaries will be added to a list
    and the tensors will be collated using
    :func:`torch.utils.data.dataloader.default_collate`.

    Args:
        items (list[BatchItem]): list of the items in the batch

    Returns:
        Batch: a batch made from `items`.
    """
    metadata = [item.metadata for item in items]
    tensors = default_collate([item.tensors for item in items])
    return Batch(metadata=metadata, tensors=tensors)


class Batch:
    """
    Represents a batch. Supports iteration
    (yields individual :class:`BatchItem`) and indexing. Slice
    indexing will return another :class:`Batch`.

    Attributes:
        metadata (list[dict[str, object]]): a list of dictionaries
            for each element in the batch.
            Each dictionary contains information
            about the corresponding item.
        tensors (dict[str, tensor]): a dictionary of collated tensors.
            The first dimension of each tensor will always be `B`,
             the batch size.
    """

    def __init__(self, metadata, tensors):
        self.metadata = metadata
        self.tensors = tensors

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise IndexError("Does not support slice with step")
            metadata = self.metadata[index]
            tensors = {
                name: tensor[index]
                for name, tensor in self.tensors.items()
            }
            return Batch(metadata=metadata, tensors=tensors)
        else:
            return BatchItem(
                metadata=self.metadata[index],
                tensors={
                    name: tensor[index]
                    for name, tensor in self.tensors.items()
                })

    def apply(self, function):
        """
        Apply a function to all tensors.

        Arguments:
            function: callable to be applied to all tensors.

        Returns:
            Batch: A new batch
        """
        tensors = {
            name: function(tensor)
            for name, tensor in self.tensors.items()
        }
        return Batch(metadata=self.metadata, tensors=tensors)

    def apply_(self, function):
        """
        Inplace variance of :meth:`apply`.
        """
        other = self.apply(function)
        self.tensors = other.tensors
        return self

    def cuda(self, *args, **kwargs):
        """
        Returns a new batch on GPU.
        """
        return self.apply(lambda x: x.cuda())

    def cuda_(self, *args, **kwargs):
        """
        Move the batch inplace to GPU.
        """
        return self.apply_(lambda x: x.cuda())

    def cpu(self, *args, **kwargs):
        """
        Returns a new batch on CPU.
        """
        return self.apply(lambda x: x.cpu())

    def cpu_(self, *args, **kwargs):
        """
        Move the batch inplace to CPU.
        """
        return self.apply_(lambda x: x.cpu_())
