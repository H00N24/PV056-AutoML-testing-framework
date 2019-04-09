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

    args = vars(parser.parse_args())

    files = sorted([x for x in os.listdir(args["results_dir"]) if x.endswith(".csv")])
    print("Dataset", "Classifier", "Configuration", "Accuracy", sep=";")
    for fl in files:
        if not args["pattern"].match(fl):
            continue
        print(*fl.split("_"), sep=";", end=";")
        dataframe = pd.read_csv(os.path.join(args["results_dir"], fl))
        all_results = dataframe.shape[0]
        accuracy = np.sum(dataframe["error"] != "+") / all_results
        print("{:.6f}".format(accuracy))


if __name__ == "__main__":
    main()
