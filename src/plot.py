import json
import time
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
import warnings
import sys
import getopt


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
    in_days: ("Number of input days [int]", "option") = 2,
    out_days: ("Number of output days [int]", "option") = 1,
    binned: (
        "Show the extended metrics for supplied comma separated binned FWI value range "
        "(e.g. 0,15,70) [Bool/list]",
        "option",
    ) = "0,5.2,11.2,21.3,38.0,50",
    case_study: (
        "The case-study region to use for inference: australia, california, portugal,"
        " siberia, chile, uk [Bool/str]",
        "option",
    ) = False,
    benchmark: (
        "Benchmark the FWI-Forecast data against FWI-Reanalysis [Bool]",
        "option",
    ) = False,
):
    """
    Process and print the dictionary of project wide arguments.

    :return: Dictionary containing configuration options.
    :rtype: dict
    """
    config_dict = {k: str2num(v) for k, v in locals().items()}
    return config_dict


def read_json(results_file: str) -> tuple:
    """
    Read and process a json results summary

    :param results_file: path to json results
    :type results_file: str
    :return: binned metrics and results summary
    :rtype: tuple
    """
    with open(results_file, "r") as res_file:
        # the results file is a json where each record
        # is of either binned or general
        # "mae_11.2_21.3_9": 9.765079498291016 OR
        # "val_acc_4": 0.6305782198905945,
        results = json.load(res_file)

        results_summary = {}

        acc_list = []
        mae_list = []
        mse_list = []

        metric_list_dict = {}
        metric_list_dict["acc"] = acc_list
        metric_list_dict["mae"] = mae_list
        metric_list_dict["mse"] = mse_list

        # first make a dict with keys acc, mae, mse
        # where the values are list of dicts
        # each dict being an individual record
        # storing the bin, day and value

        for key, value in results.items():
            key_split = key.split("_")
            if len(key_split) == 4:
                record = {}
                record["bin"] = (float(key_split[1]), float(key_split[2]))
                record["day"] = int(key_split[3])
                record["val"] = value

                metric_list_dict[key_split[0]].append(record)

            else:
                results_summary[key] = value
        # Use make_dict() function to transform the dict of list of dicts
        # to dicts all the way down
        return (make_dict(metric_list_dict), results_summary)


def make_dict(metric_list_dict: dict) -> dict:
    """
    Convert dict of list of dicts to dicts of dicts of dicts

    :param metric_list_dict: dict of list of dicts
    :type metric_list_dict: dict
    :return: dict of dict of dicts
    :rtype: dict
    """

    # Use any one of acc, mae, mse to get bins
    # assumes all three have the same metrics

    acc_metrics = metric_list_dict["acc"]
    bins_list = []
    for record in acc_metrics:
        temp_bin = record["bin"]
        if temp_bin not in bins_list:
            bins_list.append(temp_bin)

    binned_metrics = {}

    for key, value in metric_list_dict.items():
        metrics = {}
        for bin in bins_list:
            # create empty dicts for every bin
            metrics[bin] = {}
        for record in value:
            # add data to the empty dicts
            metrics[record["bin"]][record["day"]] = record["val"]
        binned_metrics[key] = metrics

    return binned_metrics


def single_day_plot(result, hparams, m, benchmark=None):
    """
    Plot mteric results for single day output.

    :param result: Model inference result dictionary
    :type result: dict
    :param hparams: Hyperparamters
    :type hparams: Namespace
    :param benchmark: Benchmark result dictionary, defaults to None
    :type benchmark: dict, optional
    :param m: metric
    :type m: str
    """
    bin_range = hparams.binned
    bin_labels = [
        f"({bin_range[i]}, {bin_range[i+1]}]"
        for i in range(len(bin_range))
        if i < len(bin_range) - 1
    ]
    bin_labels.append(f"({bin_range[-1]}, inf)")

    fwi_levels = ["V. Low", "Low", "Mod", "High", "V. High", "Extreme"]
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

    if benchmark:
        ax.legend()

    fig.tight_layout()
    file_name = m + "_" + time.strftime("%Y%m%d-%H%M%S")

    plt.savefig(file_name, dpi=600, bbox_inches="tight")


def multi_day_plot(result, hparams, benchmark=None, m="acc"):
    """
    Plot mteric results for single day output.

    :param result: Model inference result dictionary
    :type result: dict
    :param hparams: Hyperparamters
    :type hparams: Namespace
    :param benchmark: Benchmark result dictionary, defaults to None
    :type benchmark: dict, optional
    :param m: metric
    :type m: str, optional
    """
    bin_range = hparams.binned

    fwi_levels = ["V. Low", "Low", "Mod", "High", "V. High", "Extreme"]

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
            ax.plot(
                x,
                bench[i],
                color=color[i],
                marker="o",
                zorder=3,
                label=None if i else r"$\mathbf{FWI-Forecast}$",
            )

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
    handles.append("_")
    labels.append("FWI-forecast")
    ax.legend(
        handles, labels, bbox_to_anchor=(1, 1), loc="upper left", fontsize="x-small"
    )

    fig.tight_layout()

    file_name = m + "_" + time.strftime("%Y%m%d-%H%M%S")

    plt.savefig(file_name, dpi=600, bbox_inches="tight")


def plot(results_file, **kwargs) -> None:
    """
    Make plots. The kwargs are equivalent to commandline args

    :param results_file: json file of results
    :type results_file: str
    """
    plt.rcParams["figure.figsize"] = [20, 10]
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fontsize"] = "large"
    plt.rcParams["figure.titlesize"] = "medium"

    warnings.filterwarnings("ignore")

    hparams = Namespace(**get_hparams(**kwargs))

    result, results_summary = read_json(results_file)

    for m in ["acc", "mse", "mae"]:
        if hparams.out_days > 1:
            multi_day_plot(
                result=result[m], hparams=hparams, m=m, benchmark=None,
            )
        else:
            single_day_plot(
                result=result[m], hparams=hparams, m=m, benchmark=None,
            )


def main(argv):
    in_days = 4
    out_days = 10
    try:
        opts, args = getopt.getopt(argv, "hf:i:o:", ["file=", "in-days=", "out-days="])
    except getopt.GetoptError:
        print("plot.py -f <file> -i <in-days> -o <out-days>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("plot.py -f <file> -i <in-days> -o <out-days>")
            sys.exit(2)
        elif opt in ("-f", "--file"):
            file_name = str(arg)
        elif opt in ("-i", "--in-days"):
            in_days = int(arg)
        else:
            out_days = int(arg)
    config = dict(in_days=in_days, out_days=out_days,)
    plot(results_file=file_name, **config, benchmark=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("plot.py -f <file> -i <in-days> -o <out-days>")
    else:
        main(sys.argv[1:])
