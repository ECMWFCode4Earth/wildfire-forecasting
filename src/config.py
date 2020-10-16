import os
import plac
import yaml
import torch
from argparse import Namespace
from pytorch_lightning import _logger as log
import time
import sys


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
    init_features: ("Architecture complexity [int]", "option") = 20,
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
    config_dict = {k: str2num(v) for k, v in locals().items()}
    for k, v in config_dict.items():
        log.info(f" |{k.replace('_', '-'):>20} -> {str(v):<20}")
    return config_dict


def create_config():
    """
    Generates config file in yaml format from parsed arguments and saves to src/config/ dir.

    :return: hparams in Namespace
    :rtype: Namespace
    """
    parsed_args = plac.call(get_hparams, eager=False)

    # Converting dictionary to namespace
    hparams = Namespace(**parsed_args)

    # Determine path of new config file : root/src/config/deepfwi-config-YYYYMMDD-HHMMSS.yaml (UTC)
    SRC_DIR = os.path.dirname(os.path.realpath('__file__'))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    config_file_name = "deepfwi-config-"+timestr+".yaml"
    config_file_path = os.path.join(SRC_DIR, 'src', 'config', config_file_name)
    config_file_path = os.path.abspath(os.path.realpath(config_file_path))

    print("Config file saved to", config_file_path)

    # Converting dictionary to yaml and writing to file
    with open(config_file_path, "w") as f:
        yaml.dump(parsed_args, f, default_flow_style=None, sort_keys=False)

    return hparams