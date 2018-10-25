# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import gzip
import json
from pathlib import Path
import random

from scipy.io import wavfile
import torch
from torch.nn import functional as F

from ..fondation.batch import BatchItem
from ..fondation.datasets import DatasetSubset
from ..fondation import utils


class NSynthMetadata:
    """
    NSynth metadata without the wavforms.

    Arguments:
        path (Path): path to the NSynth dataset.
            This path should contain a `examples.json` file.

    An item of the nsynth metadata dataset will contain the follow tensors:
        - instrument (LongTensor)
        - pitch (LongTensor)
        - velocity (LongTensor)
        - instrument_family (LongTensor)
        - index (LongTensor)

    Attributes:
        cardinalities (dict[str, int]): cardinality of
            instrument, instrument_family, pitch and velocity
        instruments (dict[str, int]): mapping from instrument
            name to instrument index
    """
    _json_cache = {}

    _FEATURES = ['instrument', 'instrument_family', 'pitch', 'velocity']

    def _map_velocity(self, metadata):
        velocity_mapping = {
            25: 0,
            50: 1,
            75: 2,
            100: 4,
            127: 5,
        }
        for meta in self._metadata.values():
            meta["velocity"] = velocity_mapping[meta['velocity']]

    def __init__(self, path):
        self.path = Path(path)

        # Cache the json to avoid reparsing it everytime
        if self.path in self._json_cache:
            self._metadata = self._json_cache[self.path]
        else:
            if self.path.suffix == ".gz":
                file = gzip.open(self.path)
            else:
                file = open(self.path, "rb")
            self._metadata = json.load(file)
            self._map_velocity(self._metadata)
            self._json_cache[self.path] = self._metadata

        self.names = sorted(self._metadata.keys())

        # Compute the mapping instrument_name -> instrument id
        self.instruments = {}
        for meta in self._metadata.values():
            self.instruments[meta["instrument_str"]] = meta["instrument"]

        # Compute the cardinality for the features velocity, instrument,
        # pitch and instrument_family
        self.cardinalities = {}
        for feature in self._FEATURES:
            self.cardinalities[feature] = 1 + max(
                i[feature] for i in self._metadata.values())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if hasattr(index, "item"):
            index = index.item()
        name = self.names[index]
        metadata = self._metadata[name]
        tensors = {}

        metadata['name'] = name
        metadata['index'] = index
        for feature in self._FEATURES:
            tensors[feature] = torch.LongTensor([metadata[feature]])

        return BatchItem(metadata=metadata, tensors=tensors)


class NSynthDataset:
    """
    NSynth dataset.

    Arguments:
        path (Path): path to the NSynth dataset.
            This path should contain a `examples.json` file
            and an `audio` folder containing the wav files.
        pad (int): amount of padding to add to the waveforms.

    Items from this dataset will contain all the information
    coming from :class:`NSynthMetadata` as well as a `'wav'`
    tensor containing the waveform.

    Attributes:
        metadata (NSynthMetadata): metadata only dataset
    """

    def __init__(self, path, pad=0):
        self.metadata = NSynthMetadata(Path(path) / "examples.json")
        self.pad = pad

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        item = self.metadata[index]

        path = self.metadata.path.parent / "audio" / "{}.wav".format(
            item.metadata['name'])
        item.metadata['path'] = path

        _, wav = wavfile.read(str(path), mmap=True)
        wav = torch.as_tensor(wav, dtype=torch.float)
        wav /= 2**15 - 1
        item.tensors['wav'] = F.pad(wav, (self.pad, self.pad))

        return item


def make_datasets(dataset, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Take the original NSynth training dataset and split it into
    a train, valid and test set making sure that for a given instrument,
    a pitch is present in only one dataset (each pair of instrument and pitch
    has multiple occurences, one for each velocity).
    """

    per_pitch_instrument = defaultdict(list)

    if isinstance(dataset, NSynthDataset):
        metadata = dataset.metadata
    elif isinstance(dataset, NSynthMetadata):
        metadata = dataset
    else:
        raise ValueError(
            "Invalid dataset {}, should be an instance of "
            "either NSynthDataset or NSynthMetadata.".format(dataset))

    for index in range(len(metadata)):
        item = metadata[index]
        per_pitch_instrument[(item.metadata['instrument'],
                              item.metadata['pitch'])].append(index)

    with utils.random_seed_manager(random_seed):
        train = []
        valid = []
        test = []
        for indexes in per_pitch_instrument.values():
            score = random.random()
            if score < valid_ratio:
                valid.extend(indexes)
            elif score < valid_ratio + test_ratio:
                test.extend(indexes)
            else:
                train.extend(indexes)

        return DatasetSubset(dataset, train), DatasetSubset(
            dataset, valid), DatasetSubset(dataset, test)


def get_metadata_path():
    """
    Get the path to the nsynth-train metadata included with SING.
    """
    return Path(__file__).parent / "examples.json.gz"


def get_nsynth_metadata():
    return NSynthMetadata(get_metadata_path())
