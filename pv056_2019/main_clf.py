import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Process, Queue

from pv056_2019.classifiers import ClassifierManager
from pv056_2019.schemas import RunClassifiersCongfigSchema


def _valid_config_path(path):
    import argparse

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Invalid path to config file.")
    else:
        return path


def weka_worker(queue):
    while not queue.empty():
        args = queue.get()
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(";".join([args[16], args[6], args[8]]), flush=True)


def main():
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument(
        "-c",
        "--config-clf",
        type=_valid_config_path,
        help="path to classifiers config file",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--datasets-csv",
        type=_valid_config_path,
        help="Path to csv with data files",
        required=True,
    )
    args = parser.parse_args()

    with open(args.config_clf, "r") as config_file:
        conf = RunClassifiersCongfigSchema(**json.load(config_file))

    datasets = []
    with open(args.datasets_csv, "r") as datasets_csv_file:
        reader = csv.reader(datasets_csv_file, delimiter=",")
        datasets = sorted([row for row in reader], key=lambda x: os.path.getsize(x[0]))

    clf_man = ClassifierManager(conf.output_folder, conf.weka_jar_path)

    queue = Queue()
    clf_man.fill_queue_and_create_configs(queue, conf.classifiers, datasets)

    pool = [Process(target=weka_worker, args=(queue,)) for _ in range(conf.n_jobs)]

    try:
        [process.start() for process in pool]
        [process.join() for process in pool]
    except KeyboardInterrupt:
        [process.terminate() for process in pool]
        print("\nInterupted!", flush=True, file=sys.stderr)

    print("Done")


if __name__ == "__main__":
    main()
