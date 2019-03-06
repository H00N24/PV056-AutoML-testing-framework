#!/usr/bin/env python
import json
from pv056_2019.schemas import OutlierDataSchema
from pv056_2019.data_loader import DataLoader
from typing import List
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Script enriches datasets with new "
        "values based on outlier detection methods."
    )
    parser.add_argument("--config-file", "-c", required=True, help="JSON configuration")
    parser.add_argument(
        "--datasets-config-file",
        "-d",
        required=True,
        help="Filename of datasets config",
    )

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = OutlierDataSchema(**json.load(json_file))

    datasets_config: List[str] = []
    data_loader = DataLoader(conf.data_paths)
    for dataframe in data_loader.load_files():
        dataframe.apply_outlier_detectors(conf.detectors)
        file_name = dataframe._arff_data["relation"] + "_enh.arff"
        save_path = os.path.join(conf.output_dir, file_name)
        datasets_config.append(save_path)
        dataframe.arff_dump(save_path)
        break

    with open(args["datasets_config_file"], "w") as out_config:
        json.dump(datasets_config, out_config, indent=2)


if __name__ == "__main__":
    main()
