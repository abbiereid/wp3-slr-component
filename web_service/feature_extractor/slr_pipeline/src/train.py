import argparse
import os
from subprocess import CalledProcessError

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from slr.data.module import DataModule
from slr.models.module import Module
from slr.utils.callbacks import FinetuningCallback


def _get_callbacks(args):
    """Get PyTorch Lightning callbacks."""
    callbacks = []

    if not args.no_early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=20, mode='min'))

    if args.freeze_until_convergence:
        callbacks.append(FinetuningCallback(args.lr_after_unfreeze))

    callbacks.append(LearningRateMonitor())

    callbacks.append(ModelSummary(5))

    return callbacks


def _get_git_commit_hash():
    import subprocess
    try:
        hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print(f'Git commit hash is {hash}.')
        return hash
    except CalledProcessError:
        print('Failed to get git commit hash. We will write logs to the root of the log directory!')
        return ""


def train(args):
    # --- Initialization --- #
    pl.seed_everything(args.seed)

    module = Module(**vars(args))
    data_module = DataModule(**vars(args))

    callbacks = _get_callbacks(args)

    # --- Loading checkpoint --- #
    if args.checkpoint is not None:
        module.load_weights(args.checkpoint)

    if args.freeze_parts is not None:
        module.freeze_part(args.freeze_parts)

    # --- Logging --- #
    git_commit_hash = _get_git_commit_hash()
    log_dir = os.path.join(args.log_dir, git_commit_hash)
    logger = TensorBoardLogger(log_dir)

    # --- Training --- #
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    trainer.fit(module, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help='Path to the log directory.', required=True)
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping.')
    parser.add_argument('--freeze_parts', type=str, help='Comma separated list of parts to freeze.')
    parser.add_argument('--freeze_until_convergence', action='store_true',
                        help='Freeze transferred weights until the validation loss has stopped improving, then unfreeze them.')
    parser.add_argument('--lr_after_unfreeze', type=float, help='Learning rate for fine-tuning after unfreezing.')
    parser.add_argument('--seed', type=int, help='Random seed.', default=42)
    parser.add_argument('--checkpoint', type=str, help='Load weights from a pre-trained checkpoint')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Module.add_model_specific_args(parser)
    parser = DataModule.add_datamodule_specific_args(parser)

    args = parser.parse_args()

    train(args)
