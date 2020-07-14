"""
Primary training and evaluation script. Run ``python3 train.py -h`` to see available
options.
"""
import os
from argparse import Namespace
import random
from datetime import datetime
from glob import glob
import shutil
import importlib
import plac
import time
import signal

import numpy as np
import torch

import pytorch_lightning as pl

# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb

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
    Main training routine specific for this project

    :param hparams: Namespace containing configuration values
    :type hparams: Namespace
    """

    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    # Prepare model and link it with the data
    model = get_model(hparams)

    # Categorize logging
    name = hparams.model + "-" + hparams.out

    # Callback to save checkpoint of best performing model
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=f"model/checkpoints/{name}/",
        monitor="val_loss",
        verbose=True,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        period=1,
        prefix="-".join(
            [
                str(x)
                for x in (
                    name,
                    hparams.in_days,
                    hparams.out_days,
                    datetime.now().strftime("-%m/%d-%H:%M"),
                )
            ]
        ),
    )

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    # Enable logging only during training
    if not hparams.dry_run:
        # tb_logger = TensorBoardLogger(save_dir="logs/tb_logs/", name=name)
        # tb_logger.experiment.add_graph(model, model.data[0][0].unsqueeze(0))
        wandb_logger = WandbLogger(
            name=hparams.comment if hparams.comment else time.ctime(),
            project=name,
            save_dir="logs",
        )
        # if not hparams.test:
        #     wandb_logger.watch(model, log="all", log_freq=100)
        wandb_logger.log_hyperparams(model.hparams)
        for file in [
            i
            for s in [glob(x) for x in ["*.py", "dataloader/*.py", "model/*.py"]]
            for i in s
        ]:
            shutil.copy(file, wandb.run.dir)

    # ------------------------
    # INIT TRAINER
    # ------------------------

    trainer = pl.Trainer(
        auto_lr_find=False,
        # progress_bar_refresh_rate=0,
        # Profiling the code to find bottlenecks
        # profiler=pl.profiler.AdvancedProfiler('profile'),
        max_epochs=hparams.epochs if not hparams.dry_run else 1,
        # CUDA trick to speed up training after the first epoch
        benchmark=True,
        deterministic=False,
        # Sanity checks
        # fast_dev_run=False,
        # overfit_pct=0.01,
        gpus=hparams.gpus,
        precision=16 if hparams.use_16bit and hparams.gpus else 32,
        # Alternative method for 16-bit training
        # amp_level="O2",
        logger=None if hparams.dry_run else [wandb_logger],  # , tb_logger],
        checkpoint_callback=None if hparams.dry_run else checkpoint_callback,
        # Using maximum GPU memory. NB: Learning rate should be adjusted according to
        # the batch size
        # auto_scale_batch_size='binsearch',
    )

    # ------------------------
    # LR FINDER
    # ------------------------

    if hparams.find_lr:
        # Run learning rate finder
        lr_finder = trainer.lr_find(model)

        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.learning_rate = new_lr

    # ------------------------
    # BATCH SIZE SEARCH
    # ------------------------

    if hparams.search_bs:
        # Invoke the batch size search using a sophisticated algorithm.
        new_batch_size = trainer.scale_batch_size(
            model, mode="binary", steps_per_trial=50, init_val=1, max_trials=10
        )

        # Override old batch size
        model.hparams.batch_size = new_batch_size

    # ------------------------
    # 3 START TRAINING
    # ------------------------

    # Interrupt training anytime and continue to test
    signal.signal(signal.SIGINT or 255, trainer.test)

    trainer.fit(model)
    trainer.test()


def get_model(hparams):
    """
    Prepare model and the data.

    :param hparams: Holds configuration values.
    :type hparams: Namespace
    :raises ImportError: The requested model and prediction data must be compatible.
    :return: Model with the linked data.
    :rtype: Model
    """
    Model = importlib.import_module(f"model.{hparams.model}").Model
    if hparams.model in ["unet"]:
        if hparams.out == "fwi_forecast":
            ModelDataset = importlib.import_module(
                f"dataloader.{hparams.out}"
            ).ModelDataset
    elif hparams.model in [
        "unet_downsampled",
        "unet_snipped",
        "unet_tapered",
    ]:
        if hparams.out == "fwi_reanalysis":
            ModelDataset = importlib.import_module(
                f"dataloader.{hparams.out}"
            ).ModelDataset
    elif hparams.model in ["unet_interpolated"]:
        if hparams.out == "gfas_frp":
            ModelDataset = importlib.import_module(
                f"dataloader.{hparams.out}"
            ).ModelDataset
    else:
        raise ImportError(f"{hparams.model} and {hparams.out} combination invalid.")

    model = Model(hparams)
    model.prepare_data(ModelDataset)
    return model


def str2num(s):
    """
    Converts parameter strings to appropriate types.

    :param s: Parameter value
    :type s: str
    :return: Appropriately converted value
    :rtype: Varying
    """
    if isinstance(s, bool):
        return s
    s = str(s)
    if "." in s or "e-" in s:
        try:
            return float(s)
        except:
            pass
    if s.isdigit():
        return int(s)
    elif s == "None":
        return None
    else:
        if s == "True":
            return True
        elif s == "False":
            return False
    return s


def get_hparams(
    #
    # U-Net config
    init_features: ("Architecture complexity", "option") = 10,
    in_days: ("Number of input days", "option") = 2,
    out_days: ("Number of output days", "option") = 1,
    #
    # General
    epochs: ("Number of training epochs", "option") = 100,
    learning_rate: ("Maximum learning rate", "option") = 1e-3,
    loss: ("Loss function: mae, mse", "option") = "mse",
    batch_size: ("Batch size of the input", "option") = 1,
    split: ("Test split fraction", "option") = 0.2,
    use_16bit: ("Use 16-bit precision for training (train only)", "option") = True,
    gpus: ("Number of GPUs to use", "option") = 1,
    optim: (
        "Learning rate optimizer: one_cycle or cosine (train only)",
        "option",
    ) = "one_cycle",
    dry_run: ("Use small amount of data for sanity check", "option") = False,
    find_lr: ("Automatically search for an ideal learning rate", "option") = False,
    search_bs: ("Scale the batch dynamically for full GPU usage") = False,
    case_study: (
        "Limit the analysis to Australian region (inference only)",
        "option",
    ) = False,
    clip_fwi: (
        "Limit the analysis to the datapoints with 0.5 < fwi < 60 (inference only)",
        "option",
    ) = False,
    #
    # Run specific
    model: (
        "Model to use: unet, unet_downsampled, unet_snipped, unet_tapered,"
        " unet_interpolated",
        "option",
    ) = "unet_interpolated",
    out: (
        "Output data for training: fwi_forecast or fwi_reanalysis or gfas_frp",
        "option",
    ) = "gfas_frp",
    forecast_dir: (
        "Directory containing the forecast data. Alternatively set $FORECAST_DIR",
        "option",
    ) = os.environ.get("FORECAST_DIR", os.getcwd()),
    forcings_dir: (
        "Directory containing the forcings data Alternatively set $FORCINGS_DIR",
        "option",
    ) = os.environ.get("FORCINGS_DIR", os.getcwd()),
    reanalysis_dir: (
        "Directory containing the reanalysis data. Alternatively set $REANALYSIS_DIR.",
        "option",
    ) = os.environ.get("REANALYSIS_DIR", os.getcwd()),
    frp_dir: (
        "Directory containing the FRP data. Alternatively set $FRP_DIR.",
        "option",
    ) = os.environ.get("FRP_DIR", os.getcwd()),
    mask: (
        "File containing the mask stored as the numpy array",
        "option",
    ) = "dataloader/mask_frp.npy",
    test_set: (
        "Load test-set filenames from specified file instead of random split",
        "option",
    ) = False,
    thresh: ("Threshold for accuracy: Half of output MAD", "option") = 0.232943,
    comment: ("Used for logging", "option") = "FRP 0 clipping, box cox",
    #
    # Test run
    save_test_set: (
        "Save the test-set file names to the specified filepath",
        "option",
    ) = "dataloader/test_set_frp.pkl",
    checkpoint_file: ("Path to the test model checkpoint", "option",) = "",
):
    """
    Process and print the dictionary of project wide arguments.

    :return: Dictionary containing configuration options.
    :rtype: dict
    """
    d = {k: str2num(v) for k, v in locals().items()}
    for k, v in d.items():
        print(f" |{k.replace('_', '-'):>15} -> {str(v):<15}")
    return d


if __name__ == "__main__":
    """
    Script entrypoint.
    """

    # Converting dictionary to namespace
    hyperparams = Namespace(**plac.call(get_hparams, eager=False))
    # ---------------------
    # RUN TRAINING
    # ---------------------

    main(hyperparams)
