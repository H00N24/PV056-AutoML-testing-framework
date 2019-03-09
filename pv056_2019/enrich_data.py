#!/usr/bin/env python
import json
from pv056_2019.schemas import OutlierDataSchema
from pv056_2019.data_loader import DataLoader
from typing import List, Tuple
import os
import argparse
from hashlib import md5
from pv056_2019.outlier_detection import DETECTORS


def main():
    parser = argparse.ArgumentParser(
        description="Script enriches datasets with new "
        + "values based on outlier detection methods. "
        + "Available outlier detection methods: "
        + ", ".join([x for x in DETECTORS])
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

    hex_name = md5(json.dumps(conf.detectors).encode("UTF-8")).hexdigest()

    ol_config_path = os.path.join(conf.output_dir, hex_name) + ".json"

    with open(ol_config_path, "w") as ol_config:
        json.dump(conf.detectors, ol_config)

    datasets_config: List[Tuple[str, str]] = []
    data_loader = DataLoader(conf.data_paths)
    for dataframe in data_loader.load_files():
        print("Processing:", dataframe._arff_data["relation"])
        dataframe.apply_outlier_detectors(conf.detectors)
        file_name = dataframe._arff_data["relation"] + "_" + hex_name + ".arff"
        file_save_path = os.path.join(conf.output_dir, file_name)
        # save_path =
        datasets_config.append((file_save_path, ol_config_path))
        dataframe.arff_dump(file_save_path)
        break

    with open(args["datasets_config_file"], "w") as out_config:
        json.dump(datasets_config, out_config, indent=2)


if __name__ == "__main__":
    main()
