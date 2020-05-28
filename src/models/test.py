"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
from argparse import Namespace
import random
import time

import numpy as np
import torch

import pytorch_lightning as pl
from custom import Model


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

SEED = 2334
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT MODEL
    # ------------------------
    torch.set_default_dtype(torch.float32)
    wandb_logger = WandbLogger(name='Test run', project="Custom")

    model = Model.load_from_checkpoint('./_ckpt_epoch_4.ckpt')
    model.eval()
#    print(model)

    model.prepare_data()
    model.train_dataloader()
    model.val_dataloader()
    model.test_dataloader()
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        precision=16 if hparams.use_16bit else 32,
        logger=wandb_logger,
        amp_level="O2",
#        auto_scale_batch_size='binsearch',
    )

    wandb_logger.watch(model, log="parameters", log_freq=100)
    wandb_logger.log_hyperparams(hparams)
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.test(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    if 0:
        root_dir = os.path.dirname(os.path.realpath(__file__))
        parent_parser = ArgumentParser(add_help=False)

        # gpu args
        parent_parser.add_argument("--gpus", type=int, default=2, help="how many gpus")
        parent_parser.add_argument(
            "--distributed_backend",
            type=str,
            default="dp",
            help="supports three options dp, ddp, ddp2",
        )
        parent_parser.add_argument(
            "--use_16bit",
            dest="use_16bit",
            action="store_true",
            help="if true uses 16 bit precision",
        )

        # each LightningModule defines arguments relevant to it
        parser = Model.add_model_specific_args(parent_parser, root_dir)
        hyperparams = parser.parse_args()

    else:
        f = 3
        x = (269 + f - 1) // f
        y = (183 + f - 1) // f
        params = dict(
            in_width=x,
            in_length=y,
            in_depth=7,
            output_size=x * y,
            drop_prob=0.5,
            learning_rate=0.001,
            conv1={"stride": 1, "kernel_size": 3, "channels": 64},
            conv2={"stride": 1, "kernel_size": 3, "channels": 128},
            conv3={"stride": 1, "kernel_size": 3, "channels": 256},
            conv4={"stride": 2, "kernel_size": 4, "channels": 512},
            pool1={"stride": 2, "kernel_size": 2},
            pool2={"stride": 2, "kernel_size": 2},
            fc1={"out_features": int(1 * x * y)},
            fc2={"out_features": int(1 * x * y)},
            fc3={"out_features": int(1 * x * y)},
            root_dir="/root/net/",
            epochs=20,
            optimizer_name="adam",
            batch_size=3,
            split=0.2,
            use_16bit=False,
            gpus=1,
        )
        hyperparams = Namespace(**params)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
