import argparse
import csv
import json
import os
import sys

from sklearn.model_selection import KFold

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import SplitterSchema


def main():
    parser = argparse.ArgumentParser(
        description="Script splits datasets for cross-validation"
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument(
        "--datasets-file",
        "-d",
        required=True,
        help="Filename of output datasets config",
    )

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = SplitterSchema(**json.load(json_file))

    data_loader = DataLoader(conf.data_path)

    datasets_output = []
    try:
        for dataframe in data_loader.load_files():
            print("Splitting:", dataframe._arff_data["relation"], flush=True)
            dataframe = dataframe.add_index_column()
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for index, data_fold in enumerate(kfold.split(dataframe.index.values)):
                train_index, test_index = data_fold

                train_frame = dataframe.select_by_index(train_index)
                train_name = (
                    dataframe._arff_data["relation"] + "_" + str(index) + "_train.arff"
                )
                train_split_output = os.path.join(conf.train_split_dir, train_name)
                train_frame.arff_dump(train_split_output)

                test_frame = dataframe.select_by_index(test_index)
                test_name = (
                    dataframe._arff_data["relation"] + "_" + str(index) + "_test.arff"
                )
                test_split_output = os.path.join(conf.test_split_dir, test_name)
                test_frame.arff_dump(test_split_output)

                datasets_output.append([train_split_output, test_split_output, ""])

    except KeyboardInterrupt:
        print("\nInterupted!", flush=True, file=sys.stderr)

    with open(args["datasets_file"], "w") as datasets_file:
        writer = csv.writer(datasets_file, delimiter=",")
        writer.writerows(datasets_output)

    print("Done")


if __name__ == "__main__":
    main()
