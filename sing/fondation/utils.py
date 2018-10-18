# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import hashlib
from pathlib import Path
import random
import requests
import sys

import torch
import tqdm


@contextlib.contextmanager
def random_seed_manager(seed):
    """
    Context manager that will save the python RNG state,
    set the seed to `seed` and on exit, set back the python RNG state.
    """
    state = random.getstate()
    try:
        random.seed(seed)
        yield None
    finally:
        random.setstate(state)


def progress_iterator(iterator, divisions=100):
    """
    Wraps an iterator of known length and yield a tuple `(progress, item)`
    for each `item` in `iterator`. `progress` will be None except `divisions`
    times that are evenly spaced. When `progress` is not None
    it will contain the current percentage of items that have been seen.

    Arguments:
        iterator (iterator): source iterator, should support :func:`len`.
        divisions (int): progress will be every 1/divisions of the
            iterator length.

    Examples::
        >>> for (progress, element) in progress_iterator(range(500)):
        ...     if progress:
        ...         print("{:.0f}% done".format(progress))
    """

    length = len(iterator)
    division_width = length / divisions
    next_division = division_width
    for idx, element in enumerate(iterator):
        progress = None
        if (idx + 1) >= next_division or idx + 1 == length:
            next_division += division_width
            progress = (idx + 1) / length * 100
        yield progress, element


def unpad1d(tensor, pad):
    """
    Opposite of padding, will remove `pad` items on each side
    of the last dimension of `tensor`.

    Arguments:
        tensor (tensor): tensor to unpad
        pad (int): amount of padding to remove on each side.
    """
    if pad > 0:
        return tensor[..., pad:-pad]
    return tensor


def load_checkpoint(path):
    """
    Arguments:
        path (str or Path): path to load
    Returns:
        (int, object): returns a tuple (epoch, state).
    """
    if path is None or not Path(path).exists():
        return -1, None
    return torch.load(path)


def save_checkpoint(path, epoch, state):
    """
    Save a new checkpoint. A temporary file is created
    and then renamed to the target path.

    Arguments:
        path (str or Path): path to write to
        epoch (int): current epoch
        state (object): state to save
    """
    if path is None:
        return
    path = Path(path)
    tmp_path = path.parent / (path.name + ".tmp")

    torch.save((epoch, state), str(tmp_path))
    tmp_path.rename(path)


def download_file(target, url, sha256=None):
    """
    Download a file with a progress bar.

    Arguments:
        target (Path): target path to write to
        url (str): url to download
        sha256 (str or None): expected sha256 hexdigest of the file
    """
    response = requests.get(url, stream=True)
    total_length = int(response.headers.get('content-length', 0))

    if sha256 is not None:
        sha = hashlib.sha256()
        update = sha.update
    else:
        update = lambda x: None

    with tqdm.tqdm(total=total_length, unit="B", unit_scale=True) as bar:
        with open(target, "wb") as output:
            for data in response.iter_content(chunk_size=4096):
                output.write(data)
                update(data)
                bar.update(len(data))
    if sha256 is not None:
        signature = sha.hexdigest()
        if sha256 != signature:
            target.unlink()
            raise ValueError("Invalid sha256 signature when downloading {}. "
                             "Expected {} but got {}".format(
                                 url, sha256, signature))


def fatal(message, error_code=1):
    """
    Print `message` to stderr and exit with the code `error_code`.
    """
    print(message, file=sys.stderr)
    sys.exit(1)
