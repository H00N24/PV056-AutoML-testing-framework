import argparse
import os
from pv056_2019.classifiers import ClassifierManager
from pv056_2019.utils import load_config_clf, load_config_data


def _valid_config_path(path):
    import argparse

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Invalid path to config file.")
    else:
        return path


def main():
    LOGS_FOLDER = "logs/"

    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-cc",
        "--config-clf",
        type=_valid_config_path,
        help="path to classifiers config file",
        required=True,
    )
    parser.add_argument(
        "-cd",
        "--config-data",
        type=_valid_config_path,
        help="path to datasets config file",
        required=True,
    )
    args = parser.parse_args()

    weka_jar_path, classifiers = load_config_clf(args.config_clf)
    dataset_paths = load_config_data(args.config_data)

    clf_man = ClassifierManager(LOGS_FOLDER, weka_jar_path)
    clf_man.run(classifiers, dataset_paths)


if __name__ == "__main__":
    main()
