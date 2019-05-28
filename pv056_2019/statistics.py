import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd


def compile_reg(s):
    try:
        return re.compile(s)
    except Exception as e:
        print("Regex error:", e, file=sys.stderr)
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Script for counting basic statistic (Accuracy, )"
    )
    parser.add_argument(
        "--results-dir", "-r", required=True, help="Directory with results in .csv"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=compile_reg,
        default=".*",
        help="Regex for filename (Python regex)",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw data (without aggregation by dataset split)",
    )

    args = vars(parser.parse_args())

    reg = re.compile(r"removed-0*")
    files = sorted([x for x in os.listdir(args["results_dir"]) if x.endswith(".csv")])

    config_file_paths = [
        x for x in os.listdir(args["results_dir"]) if x.endswith(".json")
    ]

    config_dict = {}
    for config_file_path in config_file_paths:
        with open(os.path.join(args["results_dir"], config_file_path)) as config_file:
            basename = os.path.basename(config_file_path)

            conf_hash = basename.split("_")[1].replace(".json", "")

            config_dict[conf_hash] = json.load(config_file)

    headers = [
        "Dataset",
        "Split",
        "Classifier",
        "Outlier detection",
        "Removed",
        "Configuration",
        "Accuracy",
    ]
    data = []
    for fl in files:
        if not args["pattern"].match(fl):
            continue

        file_split = fl.split("_")
        file_split[-1] = file_split[-1].replace(".csv", "")

        if "removed-" in file_split[-1]:
            file_split[-1] = reg.sub("", file_split[-1])
        else:
            file_split.append(0)

        datest, split, classifier, conf_hash, removed = file_split

        classifier = config_dict[conf_hash]["model_config"].get("class_name")

        od_name = config_dict[conf_hash]["ad_config"].get("name", "")

        dataframe = pd.read_csv(os.path.join(args["results_dir"], fl))
        all_results = dataframe.shape[0]
        accuracy = np.sum(dataframe["error"] != "+") / all_results

        data.append([datest, split, classifier, od_name, removed, conf_hash, accuracy])

    if not args["raw"]:
        dataframe = pd.DataFrame(data, columns=headers)
        aggregated_frame = dataframe.groupby(
            ["Dataset", "Classifier", "Outlier detection", "Removed", "Configuration"]
        ).mean()
        aggregated_frame = aggregated_frame.loc[:, aggregated_frame.columns != "Split"]
        print(aggregated_frame.to_csv())
    else:
        dataframe = pd.DataFrame(data, columns=headers)
        print(dataframe.to_csv(index=False, header=False))


if __name__ == "__main__":
    main()
