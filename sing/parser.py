# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from pathlib import Path


def get_parser():
    """
    Returns:
        argparse.ArgumentParser: parser with all the options
            for the training of a SING model.
    """
    parser = argparse.ArgumentParser(
        "sing.train",
        description="Train a SING model on the NSynth dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Datasets arguments
    parser.add_argument(
        "--data",
        default="data/nsynth-train",
        type=Path,
        help="path to the dataset, e.g. .../nsynth-train")
    parser.add_argument(
        "--pad",
        type=int,
        default=2304,
        help="Extra padding added to the waveforms",
    )

    # Loss arguments
    parser.add_argument("--wav", action="store_true", help="Use a Wav loss")
    parser.add_argument(
        "--epsilon",
        default=1,
        type=float,
        help="Offset for power spectrum before taking the log")
    parser.add_argument(
        "--l1",
        action="store_true",
        help="Use L1 loss instead of mse",
    )

    # Misc arguments
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--parallel", action="store_true", help="Use multiple gpus")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the checkpoint folder")
    parser.add_argument(
        "--output",
        type=Path,
        default="models/sing.th",
        help="Path to output final SING model")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Debug flag")
    parser.add_argument(
        "-f", "--debug-fast", action="store_true", help="Debug fast flag")

    # Common arguments
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="Learning rate for Adam")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size")

    # Autoencoder arguments
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=50,
        help="Number of epochs for the autoencoder")

    parser.add_argument(
        "--ae-channels",
        type=int,
        default=4096,
        help="Number of channels in the autoencoder")
    parser.add_argument(
        "--ae-stride", type=int, default=256, help="Stride of the autoencoder")
    parser.add_argument(
        "--ae-dimension",
        type=int,
        default=128,
        help="Dimension of the autoencoder embedding")
    parser.add_argument(
        "--ae-kernel",
        type=int,
        default=1024,
        help="Kernel size of the autoencoder")
    parser.add_argument(
        "--ae-rewrite",
        type=int,
        default=2,
        help="Number of rewrite layers in the autoencoder")
    parser.add_argument(
        "--ae-context",
        type=int,
        default=9,
        help="Context size of the decoder")
    parser.add_argument(
        "--ae-window",
        default="hann",
        help="Window to use to smooth convolutions. Default to 'hann'. "
        "To deactivate, use --ae-no-window")
    parser.add_argument(
        "--ae-no-window", dest="ae_window", action="store_const", const=None)
    parser.add_argument(
        "--ae-squared-window",
        action="store_true",
        default=True,
        help="Square the window used to smooth convolutions. "
        "To deactivate, use --ae-no-squared-window.")
    parser.add_argument(
        "--ae-no-squared-window",
        action="store_false",
        dest="ae_squared_window")

    # Sequence generator arguments
    parser.add_argument(
        "--seq-hidden-size",
        type=int,
        default=1024,
        help="Size of the LSTM hidden layers")
    parser.add_argument(
        "--seq-layers",
        type=int,
        default=3,
        help="Number of layers in the LSTM")
    parser.add_argument(
        "--seq-epochs",
        type=int,
        default=50,
        help="Number of epochs for the sequence generator")
    parser.add_argument(
        "--seq-truncated",
        type=int,
        default=32,
        help="Truncated gradient for the sequence generator. "
        "0 means using the full sequence.")
    parser.add_argument(
        "--sing-epochs",
        type=int,
        default=20,
        help="Number of fine tuning epochs for the full SING model")

    # Lookup tables arguments
    parser.add_argument(
        "--time-dim",
        type=int,
        default=4,
        help="Dimension of the time step lookup table")
    parser.add_argument(
        "--instrument-dim",
        type=int,
        default=16,
        help="Dimension of the instrument embedding")
    parser.add_argument(
        "--pitch-dim",
        type=int,
        default=8,
        help="Dimension of the pitch embedding")
    parser.add_argument(
        "--velocity-dim",
        type=int,
        default=2,
        help="Dimension of the velocity embedding")
    return parser
