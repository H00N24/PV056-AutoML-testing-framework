#!/usr/bin/env python
import argparse
import json
import os
import sys
from hashlib import md5
from typing import List, Tuple

from pv056_2019.data_loader import DataLoader
from pv056_2019.outlier_detection import DETECTORS
from pv056_2019.schemas import OutlierDataSchema


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
        help="Filename of output datasets config",
    )

    args = vars(parser.parse_args())

    with open(args["config_file"]) as json_file:
        conf = OutlierDataSchema(**json.load(json_file))

    hex_name = md5(
        json.dumps(conf.detectors, sort_keys=True).encode("UTF-8")
    ).hexdigest()

    ol_config_path = os.path.join(conf.output_dir, hex_name) + ".json"

    with open(ol_config_path, "w") as ol_config:
        json.dump(conf.detectors, ol_config)

    datasets_config: List[Tuple[str, str]] = []
    data_loader = DataLoader(conf.data_path)
    for dataframe in data_loader.load_files():
        try:
            print("Processing:", dataframe._arff_data["relation"], flush=True)
            dataframe.apply_outlier_detectors(conf.detectors)
            file_name = dataframe._arff_data["relation"] + "_" + hex_name + ".arff"
            file_save_path = os.path.join(conf.output_dir, file_name)
            # save_path =
            datasets_config.append((file_save_path, ol_config_path))
            dataframe.arff_dump(file_save_path)
        except KeyboardInterrupt:
            print("\nInterupted!", flush=True, file=sys.stderr)
            break

    print("Saving datasets config file.")
    with open(args["datasets_config_file"], "w") as out_config:
        json.dump(datasets_config, out_config, indent=2)

    print("Done")


if __name__ == "__main__":
    main()
