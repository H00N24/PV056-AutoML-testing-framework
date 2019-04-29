import argparse
import csv
import json
import os
import sys
from hashlib import md5

from pv056_2019.data_loader import DataLoader
from pv056_2019.schemas import ODStepConfigSchema


def main():
    parser = argparse.ArgumentParser(
        description="Apply outlier detection methods to training data"
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
        conf = ODStepConfigSchema(**json.load(json_file))

    train_data_loader = DataLoader(conf.train_split_dir, regex=r".*_train\.arff")
    test_data_loader = DataLoader(conf.test_split_dir, regex=r".*_test\.arff")

    test_data_dict = {
        os.path.basename(path).replace("_test.arff", ""): path
        for path in test_data_loader.file_paths
    }
    datasets_output = []

    try:
        for od_settings in conf.od_methods:
            hex_name = md5(od_settings.json(sort_keys=True).encode("UTF-8")).hexdigest()
            print("Applying", od_settings.name, "(" + hex_name + ")")
            config_save_path = os.path.join(conf.train_od_dir, hex_name + ".json")
            with open(config_save_path, "w") as out_config:
                out_config.write(od_settings.json(sort_keys=True))

            for dataframe, train_file_path in zip(
                train_data_loader.load_files(), train_data_loader.file_paths
            ):
                od_frame = dataframe.apply_outlier_detector(od_settings)

                file_basename = os.path.basename(train_file_path)
                print("   ", file_basename)
                file_name = file_basename.replace(
                    "_train.arff", "_" + hex_name + "_train.arff"
                )
                file_save_path = os.path.join(conf.train_od_dir, file_name)

                od_frame.arff_dump(file_save_path)

                datasets_output.append(
                    [
                        file_save_path,
                        test_data_dict[file_basename.replace("_train.arff", "")],
                        config_save_path,
                    ]
                )
    except KeyboardInterrupt:
        print("\nInterupted!", flush=True, file=sys.stderr)

    with open(args["datasets_file"], "w") as datasets_file:
        writer = csv.writer(datasets_file, delimiter=",")
        writer.writerows(datasets_output)

    print("Done")


if __name__ == "__main__":
    main()
