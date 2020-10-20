import json
from pprint import pprint


def read_json(results_file: str) -> tuple:
    """
    Read and process a json results summary

    :param results_file: path to json results
    :type results_file: str
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


metric_list_dict, results_summary = read_json("results.json")

pprint(metric_list_dict)
