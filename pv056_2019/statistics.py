import argparse
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

    headers = ["Dataset", "Split", "Classifier", "Configuration", "Removed", "Accuracy"]
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

        dataframe = pd.read_csv(os.path.join(args["results_dir"], fl))
        all_results = dataframe.shape[0]
        accuracy = np.sum(dataframe["error"] != "+") / all_results

        data.append(file_split + [accuracy])

    if not args["raw"]:
        dataframe = pd.DataFrame(data, columns=headers)
        aggregated_frame = dataframe.groupby(
            ["Dataset", "Classifier", "Configuration", "Removed"]
        ).mean()
        aggregated_frame = aggregated_frame.loc[:, aggregated_frame.columns != "Split"]
        print(aggregated_frame.to_csv())
    else:
        print(";".join(headers))
        for line in data:
            print(",".join([str(x) for x in line]))


if __name__ == "__main__":
    main()
