"""
Primary inference and testing script. Run ``python3 test.py -h`` to see available
options.
"""
from argparse import Namespace
import random
import plac
import sys
import logging
import json
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from IPython.display import clear_output

import torch

import pytorch_lightning as pl

sys.path.append("src")
from train import get_hparams, get_model  # noqa: E402

# Setting seeds to ensure reproducibility. Setting CUDA to deterministic mode slows down
# the training.
SEED = 2334
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main(hparams, verbose=True):
    """
    Main testing routine specific for this project

    :param hparams: Namespace containing configuration values
    :type hparams: Namespace
    """
    # Set the evaluation flag
    hparams.eval = True

    # ------------------------
    # 1 INIT MODEL
    # ------------------------

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    model = get_model(hparams)
    if not hparams.benchmark:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(hparams.checkpoint_file)["state_dict"])
        else:
            model.load_state_dict(torch.load(hparams.checkpoint_file, torch.device('cpu'))["state_dict"])
    model.eval()

    # ------------------------
    # LOGGING SETUP
    # ------------------------

    trainer = pl.Trainer(gpus=hparams.gpus)  # , tb_logger],

    # ------------------------
    # 3 START TESTING
    # ------------------------

    # Temporary fix until next release of pytorch-lightning
    # configured to work for pytorch_lightning 0.9.0 rc12
    try:
        result = trainer.test(model, verbose=verbose)[0]
    except:
        result = trainer.test(model)[0]
 
    with open('result.json', 'w') as outfile:
        json.dump(result, outfile)

    return result, model.hparams


def autolabel(rects, ax, width):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    :param rects: Bar containers
    :type rects: matplotlib.container.BarContainer
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=min(90 * width, 25),
            zorder=5,
        )


def single_day_plot(result, hparams, m, benchmark=None):
    """
    Plot mteric results for single day output.

    :param result: Model inference result dictionary
    :type result: dict
    :param hparams: Hyperparamters
    :type hparams: Namespace
    :param benchmark: Benchmark result dictionary, defaults to None
    :type benchmark: dict, optional
    """
    bin_range = hparams.binned
    bin_labels = [
        f"({bin_range[i]}, {bin_range[i+1]}]"
        for i in range(len(bin_range))
        if i < len(bin_range) - 1
    ]
    bin_labels.append(f"({bin_range[-1]}, inf)")
    
    fwi_levels = ['V. Low', 'Low', 'Mod', 'High', 'V. High', 'Extreme']
    bin_labels = [bin_labels[i] + "\n" + fwi_levels[i] for i in range(len(bin_labels))]
    

    xlab = "FWI Level"
    # The label locations
    x = np.arange(len(bin_labels))
    # The width of the bars
    width = 0.7 / (3 if benchmark else 2)
    title = f"{hparams.in_days} Day Input // 1 Day Prediction (Global)"
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ylabel = {
        "acc": "Accuracy",
        "mae": "Mean absolute error",
        "mse": "Mean squared error",
    }
    ax.set_ylabel(ylabel[m])
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)

    num_groups = 2 if benchmark else 1

    rect_list = []
    preds = [x[0] for x in result.values()]
    rect_list.append(
        ax.bar(
            x - width * num_groups / 2 + width * 1 / 2, preds, width, label="deepFWI"
        )
    )
    if benchmark:
        bench = [x[0] for x in benchmark.values()]
        rect_list.append(
            ax.bar(
                x - width * num_groups / 2 + width * 1 / 2 + width * 1,
                bench,
                width,
                label="FWI-Forecast",
            )
        )

    # bar labels not needed, since it is not legible
#     for rect in rect_list:
#         autolabel(rect, ax, width)

    if benchmark:
        ax.legend()

    fig.tight_layout()
    plt.show()


def multi_day_plot(result, hparams, benchmark=None, m="acc"):
    bin_range = hparams.binned
    
    fwi_levels = ['V. Low', 'Low', 'Mod', 'High', 'V. High', 'Extreme']
    
    bin_labels = [
        f"({bin_range[i]}, {bin_range[i+1]}]"
        for i in range(len(bin_range))
        if i < len(bin_range) - 1
    ]
    bin_labels.append(f"({bin_range[-1]}, inf)")
    
    bin_labels = [bin_labels[i] + " " + fwi_levels[i] for i in range(len(bin_labels))]

    labels = list(range(len(list(result.values())[0])))

    width = 1 / (len(bin_range) + 1)
    x = np.arange(len(list(result.values())[0]))

    preds = [list(x.values()) for x in result.values()]

    fig, ax = plt.subplots()
    rects = []
    
    color = ["#008000", "#FFFF00", "#FFA500", "#FF0000", "#654321", "#000000"]
    for i in range(len(bin_range)):
        rects.append(
            ax.bar(
                x - width * (len(bin_range) + 1) / 2 + (i + 1) * width,
                [x for x in preds[i]],
                width,
                label=bin_labels[i],
                align="center",
                color=color[i],
            )
        ) 
        
    
    if benchmark:
        bench = [list(x.values()) for x in benchmark.values()]
        for i in range(len(bin_range)):
            ax.plot(x, bench[i], color=color[i], marker='o', zorder=3, label=None if i else r"$\mathbf{FWI-Forecast}$")
            
    ax.plot([], [], " ", label=r"$\mathbf{deepFWI}$")

    ylabel = {
        "acc": "Accuracy in %",
        "mae": "Mean absolute error (A.U.)",
        "mse": "Mean squared error (A.U.)",
    }
    ax.set_ylabel(ylabel[m])
    ax.set_title(
        f"{hparams.in_days} Day Input // {hparams.out_days} Day Prediction "
        f"({hparams.case_study if hparams.case_study else 'Global'})"
    )

    ax.set_xticks(x)
    ax.set_xlabel("Day of FWI Prediction")
    ax.set_xticklabels(labels)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append('_')
    labels.append('FWI-forecast')
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", fontsize='x-small')

    # bar labels not needed, since it is not legible
#     for rect in rects:
#         autolabel(rect, ax, width)

    fig.tight_layout()

    plt.show()


def process_result(result):
    metrics = ["acc", "mse", "mae"]
    processed_dict = defaultdict(lambda: defaultdict(dict))
    for metric in metrics:
        for r in result:
            if metric in r:
                splits = r.split("_")
                try:
                    processed_dict[metric][(float(splits[1]), float(splits[2]))][
                        int(splits[3])
                    ] = (
                        round(result[r] * 100, 2)
                        if metric == "acc"
                        else round(result[r], 2)
                    )
                except IndexError:
                    pass
    return processed_dict


def run(**kwargs):
    """
    Run inference and do benchmarks if benchmarks=True supplied. The kwargs are \
equivalent to the commandline arguments.

    :return: Result metrics and the inferred hyperparamters
    :rtype: tuple
    """
    plt.rcParams["figure.figsize"] = [20, 10]
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fontsize"] = "large"
    plt.rcParams["figure.titlesize"] = "medium"

    logging.disable(sys.maxsize)
    warnings.filterwarnings("ignore")

    hparams = Namespace(**get_hparams(**kwargs))

    if hparams.benchmark:
        print("Doing inference with the model..")
        hparams.benchmark = False
        result, _ = main(hparams=hparams, verbose=False)
        hparams.benchmark = True

        print("Running benchmarks..")
        benchmark_result, hparams = main(hparams=hparams, verbose=False)
        benchmark_result = process_result(benchmark_result)
    else:
        benchmark_result = None
        print("Doing inference with the model..")
        result, hparams = main(hparams=hparams, verbose=False)

    clear_output(wait=True)
    result = process_result(result)

    for m in ["acc", "mse", "mae"]:
        if hparams.out_days > 1:
            multi_day_plot(
                result=result[m], hparams=hparams, m=m, benchmark=benchmark_result[m],
            )
        else:
            single_day_plot(
                result=result[m], hparams=hparams, m=m, benchmark=benchmark_result[m],
            )

    return (result, benchmark_result), hparams


if __name__ == "__main__":
    """
    Script entrypoint.
    """

    # Converting dictionary to namespace
    hparams = Namespace(**plac.call(get_hparams, eager=False))
    # ---------------------
    # RUN TESTING
    # ---------------------

    main(hparams)
