# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools

from torch import nn
import torch

from .parser import get_parser
from . import nsynth, dsp
from .ae.models import ConvolutionalAE
from .ae.trainer import AutoencoderTrainer
from .fondation import utils, datasets
from .sequence.models import SequenceGenerator, SING
from .sequence.trainer import SequenceGeneratorTrainer, SINGTrainer
from .sequence.utils import generate_embeddings_dataset


def train_autoencoder(args, **kwargs):
    checkpoint_path = args.checkpoint / "ae.torch" if args.checkpoint else None
    model = ConvolutionalAE(
        channels=args.ae_channels,
        stride=args.ae_stride,
        dimension=args.ae_dimension,
        kernel_size=args.ae_kernel,
        context_size=args.ae_context,
        rewrite_layers=args.ae_rewrite,
        window_name=args.ae_window,
        squared_window=args.ae_squared_window)
    advised_pad = model.decoder.strip + 512
    if args.pad != advised_pad:
        print("Warning, best padding for the current settings is {}, "
              "current value is {}.".format(advised_pad, args.pad))
    if args.ae_epochs:
        print("Training autoencoder")
        AutoencoderTrainer(
            suffix="_ae",
            model=model,
            epochs=args.ae_epochs,
            checkpoint_path=checkpoint_path,
            **kwargs).train()
    return model


def train_sequence_generator(args, autoencoder, cardinalities, train_dataset,
                             eval_datasets, train_loss, eval_losses, **kwargs):
    checkpoint_path = (args.checkpoint / "seq.torch"
                       if args.checkpoint else None)

    wav_length = train_dataset[0].tensors['wav'].size(-1)
    embedding_length = autoencoder.decoder.embedding_length(
        wav_length - 2 * autoencoder.decoder.strip)
    embeddings = {
        name: (cardinalities[name], getattr(args, '{}_dim'.format(name)))
        for name in ['velocity', 'instrument', 'pitch']
    }

    model = SequenceGenerator(
        embeddings=embeddings,
        length=embedding_length,
        time_dimension=args.time_dim,
        output_dimension=args.ae_dimension,
        hidden_size=args.seq_hidden_size,
        num_layers=args.seq_layers)

    if args.seq_epochs:
        print("Precomputing embeddings for all datasets")
        generate_embeddings = functools.partial(
            generate_embeddings_dataset,
            encoder=autoencoder.encoder,
            batch_size=args.batch_size,
            cuda=args.cuda,
            parallel=args.parallel)
        train_dataset = generate_embeddings(train_dataset)

        print("Training sequence generator")
        SequenceGeneratorTrainer(
            suffix="_seq",
            model=model,
            decoder=autoencoder.decoder,
            epochs=args.seq_epochs,
            train_loss=nn.MSELoss(),
            eval_losses=eval_losses,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            truncated_gradient=args.seq_truncated,
            checkpoint_path=checkpoint_path,
            **kwargs).train()
    return model


def fine_tune_sing(args, sequence_generator, decoder, **kwargs):
    print("Fine tuning SING")
    checkpoint_path = (args.checkpoint / "sing.torch"
                       if args.checkpoint else None)
    model = SING(sequence_generator=sequence_generator, decoder=decoder)

    if args.sing_epochs:
        SINGTrainer(
            suffix="_sing",
            epochs=args.sing_epochs,
            model=model,
            checkpoint_path=checkpoint_path,
            **kwargs).train()
    return model


def main():
    args = get_parser().parse_args()

    if args.debug:
        args.ae_epochs = 1
        args.seq_epochs = 1
        args.sing_epochs = 1

    if args.debug_fast:
        args.ae_channels = 128
        args.ae_dimension = 16
        args.ae_rewrite = 1
        args.seq_hidden_size = 128
        args.seq_layers = 1

    if args.checkpoint:
        args.checkpoint.mkdir(exist_ok=True, parents=True)

    if not args.data.exists():
        utils.fatal("Could not find the nsynth dataset. "
                    "To download it, follow the instructions at "
                    "https://github.com/facebookresearch/SING")

    nsynth_dataset = nsynth.NSynthDataset(args.data, pad=args.pad)
    cardinalities = nsynth_dataset.metadata.cardinalities

    train_dataset, valid, test = nsynth.make_datasets(nsynth_dataset)
    if args.debug:
        train_dataset = datasets.RandomSubset(train_dataset, size=100)
    eval_train = datasets.RandomSubset(train_dataset, size=10000)

    if args.debug:
        eval_datasets = {
            'eval_train': eval_train,
        }
    else:
        eval_datasets = {
            'eval_train': eval_train,
            'valid': valid,
            'test': test,
        }

    base_loss = nn.L1Loss() if args.l1 else nn.MSELoss()
    train_loss = base_loss if args.wav else dsp.SpectralLoss(
        base_loss, epsilon=args.epsilon)
    eval_losses = {
        'wav_l1': nn.L1Loss(),
        'wav_mse': nn.MSELoss(),
        'spec_l1': dsp.SpectralLoss(nn.L1Loss(), epsilon=args.epsilon),
        'spec_mse': dsp.SpectralLoss(nn.MSELoss(), epsilon=args.epsilon),
    }

    kwargs = {
        'train_dataset': train_dataset,
        'eval_datasets': eval_datasets,
        'train_loss': train_loss,
        'eval_losses': eval_losses,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'cuda': args.cuda,
        'parallel': args.parallel,
    }

    autoencoder = train_autoencoder(args, **kwargs)
    sequence_generator = train_sequence_generator(args, autoencoder,
                                                  cardinalities, **kwargs)
    sing = fine_tune_sing(args, sequence_generator, autoencoder.decoder,
                          **kwargs)
    torch.save(sing.cpu(), str(args.output))


if __name__ == "__main__":
    main()
