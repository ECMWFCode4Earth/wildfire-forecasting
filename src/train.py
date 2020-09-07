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
import sys

import numpy as np
import torch

import pytorch_lightning as pl

# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import _logger as log
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
        filepath=f"src/model/checkpoints/{name}/",
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
            for s in [
                glob(x) for x in ["src/*.py", "src/dataloader/*.py", "src/model/*.py"]
            ]
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
    results = trainer.test()

    return results


def set_hparams(hparams):
    """
    Add constant parameter values based on passed arguments.

    :param hparams: Parameters
    :type hparams: Namespace
    :return: Modified parameters
    :rtype: Namespace
    """

    # Check for CUDA availability if gpus > 0 requested
    if hparams.gpus and not torch.cuda.is_available():
        hparams.gpus = 0

    if hparams.benchmark:
        # Use empty model while benchmarking with fwi-forecast
        hparams.model = "base_model"
        hparams.out = "fwi_reanalysis"
        hparams.eval = True
        hparams.gpus = 0

    if hparams.case_study:
        case_studies = importlib.import_module("data.consts.case_study").case_studies
        hparams.case_study_dates = case_studies[hparams.case_study]
        hparams.mask = f"src/dataloader/mask/{hparams.case_study}_mask.npy"

    from data.consts.forcing_stats import (
        FORCING_STD_TP,
        FORCING_STD_T2,
        FORCING_STD_WSPEED,
        FORCING_STD_RH,
        FORCING_MEAN_WSPEED,
        FORCING_MEAN_TP,
        FORCING_MEAN_T2,
        FORCING_MEAN_RH,
    )

    hparams.inp_mean = {
        "wspeed": FORCING_MEAN_WSPEED,
        "tp": FORCING_MEAN_TP,
        "t2": FORCING_MEAN_T2,
        "rh": FORCING_MEAN_RH,
    }
    hparams.inp_std = {
        "wspeed": FORCING_STD_WSPEED,
        "tp": FORCING_STD_TP,
        "t2": FORCING_STD_T2,
        "rh": FORCING_STD_RH,
    }

    if hparams.smos_input:
        from data.consts.soil_moisture_stats import (
            SOIL_MOISTURE_MEAN,
            SOIL_MOISTURE_STD,
        )

        hparams.smos_mean = SOIL_MOISTURE_MEAN
        hparams.smos_std = SOIL_MOISTURE_STD

    if hparams.out == "fwi_reanalysis":
        from data.consts.fwi_reanalysis_stats import (
            REANALYSIS_FWI_MEAN,
            REANALYSIS_FWI_MAD,
            REANALYSIS_FWI_VAR,
        )

        hparams.out_mean, hparams.out_mad, hparams.out_var = (
            REANALYSIS_FWI_MEAN,
            REANALYSIS_FWI_MAD,
            REANALYSIS_FWI_VAR,
        )

    elif hparams.out == "gfas_frp":
        from data.consts.frp_stats import (
            FRP_MEAN,
            FRP_MAD,
            FRP_VAR,
            BOX_COX_FRP_MEAN,
            BOX_COX_FRP_MAD,
            BOX_COX_FRP_VAR,
            BOX_COX_LAMBDA,
        )

        hparams.out_mean, hparams.out_mad, hparams.out_var = (
            BOX_COX_FRP_MEAN if hparams.boxcox else FRP_VAR,
            BOX_COX_FRP_MAD if hparams.boxcox else FRP_MAD,
            BOX_COX_FRP_VAR if hparams.boxcox else FRP_MEAN,
        )
        if hparams.boxcox and not (type(hparams.boxcox) == type(bool)):
            hparams.boxcox = BOX_COX_LAMBDA

    if hparams.cb_loss and hparams.out == "fwi_reanalysis":
        from data.consts.reanalysis_freq import bin_centers, freq

        # Not allow zero frequency for numerical stability
        freq[freq == 0] = 1

        hparams.bin_centers = bin_centers
        hparams.loss_factors = (1 - hparams.cb_loss) / (1 - hparams.cb_loss ** freq)
        hparams.loss_factors = (
            hparams.loss_factors
            / hparams.loss_factors.sum()
            * hparams.loss_factors.size
        )
        assert (
            hparams.bin_centers.shape == hparams.loss_factors.shape
        ), "The number of bin-centers for corresponding frequencies must be the same"
    return hparams


def get_model(hparams):
    """
    Prepare model and the data.

    :param hparams: Holds configuration values.
    :type hparams: Namespace
    :raises ImportError: The requested model and prediction data must be compatible.
    :return: Model with the linked data.
    :rtype: Model
    """
    sys.path += ["../", "."]

    # Update hparams with the constants
    set_hparams(hparams)

    if hparams.model in ["base_model"]:
        Model = importlib.import_module(f"model.{hparams.model}").BaseModel
        if hparams.out == "fwi_reanalysis":
            ModelDataset = importlib.import_module(
                f"dataloader.{hparams.out}"
            ).ModelDataset
            ModelDataset.BenchmarkDataset = importlib.import_module(
                "dataloader.fwi_forecast"
            ).ModelDataset
    else:
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

    model = Model(hparams).to("cuda" if hparams.gpus else "cpu")
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
    if "," in s:
        return [str2num(i) for i in s.split(",")]
    if "." in s or "e-" in s:
        try:
            return float(s)
        except:
            pass
    elif s.isdigit():
        return int(s)
    elif s.lower() == "inf":
        return float("inf")
    elif s.lower() == "none":
        return None
    else:
        if s.lower() == "true":
            return True
        elif s.lower() == "false":
            return False
    return s


def get_hparams(
    #
    # U-Net config
    init_features: ("Architecture complexity [int]", "option") = 16,
    in_days: ("Number of input days [int]", "option") = 2,
    out_days: ("Number of output days [int]", "option") = 1,
    #
    # General
    epochs: ("Number of training epochs [int]", "option") = 100,
    learning_rate: ("Maximum learning rate [float]", "option") = 1e-3,
    batch_size: ("Batch size of the input [int]", "option") = 1,
    split: ("Test split fraction [float]", "option") = 0.2,
    use_16bit: ("Use 16-bit precision for training (train only)", "option") = True,
    gpus: ("Number of GPUs to use [int]", "option") = 1,
    optim: (
        "Learning rate optimizer: one_cycle or cosine (train only) [str]",
        "option",
    ) = "one_cycle",
    dry_run: ("Use small amount of data for sanity check [Bool]", "option") = False,
    find_lr: (
        "Automatically search for an ideal learning rate [Bool]",
        "option",
    ) = False,
    search_bs: (
        "Scale the batch dynamically for full GPU usage [Bool]",
        "option",
    ) = False,
    case_study: (
        "The case-study region to use for inference: australia, california, portugal,"
        " siberia, chile, uk [Bool/str]",
        "option",
    ) = False,
    clip_output: (
        "Limit the inference to the datapoints within supplied range (e.g. 0.5,60) "
        "[Bool/list]",
        "option",
    ) = False,
    boxcox: (
        "Apply boxcox transformation with specified lambda while training and the "
        "inverse boxcox transformation during the inference. [Bool/float]",
        "option",
    ) = 0.1182,
    binned: (
        "Show the extended metrics for supplied comma separated binned FWI value range "
        "(e.g. 0,15,70) [Bool/list]",
        "option",
    ) = "0,5.2,11.2,21.3,38.0,50",
    round_to_zero: (
        "Round off the target values below the specified threshold to zero "
        "[Bool/float]",
        "option",
    ) = False,
    isolate_frp: (
        "Exclude the isolated datapoints with FRP > 0 [Bool]",
        "option",
    ) = False,
    date_range: (
        "Filter the data with specified date range in YYYY-MM-DD format. E.g. "
        "2019-04-01,2019-05-01 "
        "[Bool/str]",
        "option",
    ) = False,
    cb_loss: (
        "Use Class-Balanced loss with the supplied beta parameter [Bool/float]",
        "option",
    ) = False,
    chronological_split: (
        "Do chronological train-test split in the specified ratio [Bool/float]",
        "option",
    ) = False,
    undersample: (
        "Undersample the datapoints with smaller than specified FWI [Bool/float]",
        "option",
    ) = False,
    #
    # Run specific
    model: (
        "Model to use: unet, unet_downsampled, unet_snipped, unet_tapered,"
        " unet_interpolated [str]",
        "option",
    ) = "unet_tapered",
    out: (
        "Output data for training: fwi_reanalysis or gfas_frp [str]",
        "option",
    ) = "fwi_reanalysis",
    benchmark: (
        "Benchmark the FWI-Forecast data against FWI-Reanalysis [Bool]",
        "option",
    ) = False,
    smos_input: ("Use soil-moisture input data [Bool]", "option") = "False",
    forecast_dir: (
        "Directory containing the forecast data. Alternatively set $FORECAST_DIR [str]",
        "option",
    ) = os.environ.get("FORECAST_DIR"),
    forcings_dir: (
        "Directory containing the forcings data Alternatively set $FORCINGS_DIR [str]",
        "option",
    ) = os.environ.get("FORCINGS_DIR"),
    smos_dir: (
        "Directory containing the soil-moisture data Alternatively set $SMOS_DIR [str]",
        "option",
    ) = os.environ.get("SMOS_DIR"),
    reanalysis_dir: (
        "Directory containing the reanalysis data. Alternatively set $REANALYSIS_DIR. "
        "[str]",
        "option",
    ) = os.environ.get("REANALYSIS_DIR"),
    frp_dir: (
        "Directory containing the FRP data. Alternatively set $FRP_DIR. [str]",
        "option",
    ) = os.environ.get("FRP_DIR"),
    mask: (
        "File containing the mask stored as the numpy array [str]",
        "option",
    ) = "src/dataloader/mask/reanalysis_mask.npy",
    comment: ("Used for logging [str]", "option") = False,
    checkpoint_file: (
        "Path to the test model checkpoint [Bool/str]",
        "option",
    ) = False,
):
    """
    Process and print the dictionary of project wide arguments.

    :return: Dictionary containing configuration options.
    :rtype: dict
    """
    d = {k: str2num(v) for k, v in locals().items()}
    for k, v in d.items():
        log.info(f" |{k.replace('_', '-'):>20} -> {str(v):<20}")
    return d


if __name__ == "__main__":
    """
    Script entrypoint.
    """

    # Converting dictionary to namespace
    hparams = Namespace(**plac.call(get_hparams, eager=False))
    # ---------------------
    # RUN TRAINING
    # ---------------------

    main(hparams)
