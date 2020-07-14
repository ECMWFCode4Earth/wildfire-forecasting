"""
Primary inference and testing script. Run ``python3 test.py -h`` to see available
options.
"""
from argparse import Namespace
import random
import time
from glob import glob
import shutil
import plac

import numpy as np
import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

from train import get_hparams, get_model

# Setting seeds to ensure reproducibility. Setting CUDA to deterministic mode slows down
# the training.
SEED = 2334
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(hparams):
    """
    Main testing routine specific for this project

    :param hparams: Namespace containing configuration values
    :type hparams: Namespace
    """

    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    model = get_model(hparams)
    model.load_state_dict(torch.load(hparams.checkpoint_file)["state_dict"])
    model.eval()

    name = "-".join([hparams.model, hparams.out, "-test"])

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    tb_logger = TensorBoardLogger(save_dir="logs/tb_logs/", name=name)
    tb_logger.experiment.add_graph(model, model.data[0][0].unsqueeze(0))
    wandb_logger = WandbLogger(
        name=hparams.comment if hparams.comment else time.ctime(),
        project=name,
        save_dir="logs",
    )
    wandb_logger.watch(model, log="all", log_freq=200)
    wandb_logger.log_hyperparams(model.hparams)
    for file in [
        i
        for s in [glob(x) for x in ["*.py", "dataloader/*.py", "model/*.py"]]
        for i in s
    ]:
        shutil.copy(file, wandb.run.dir)

    trainer = pl.Trainer(gpus=hparams.gpus, logger=[wandb_logger])  # , tb_logger],

    # ------------------------
    # 3 START TESTING
    # ------------------------

    trainer.test(model)


if __name__ == "__main__":
    """
    Script entrypoint.
    """

    # Converting dictionary to namespace
    hyperparams = Namespace(**plac.call(get_hparams, eager=False))
    # Set the evaluation flag in hyperparamters
    hyperparams.eval = True
    # ---------------------
    # RUN TESTING
    # ---------------------

    main(hyperparams)
