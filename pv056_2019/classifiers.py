import json
import os
import hashlib
import re
from multiprocessing import Queue
from pv056_2019.utils import ID_NAME, OD_VALUE_NAME

from pv056_2019.schemas import ClassifierSchema
from typing import List
from itertools import product


class ClassifierManager:

    # Weka classifiers
    # http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html

    # java -cp weka.jar weka.classifiers.meta.FilteredClassifier
    # -t data/diabetes.arff
    # -F "weka.filters.unsupervised.attribute.RemoveByName -E ^ID$"
    # -x 5 -S 1
    # -W weka.classifiers.trees.J48 -- -C 0.25 -M 2

    def __init__(self, log_folder, weka_jar_path):
        self.log_folder = log_folder
        if not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True)
        self.weka_jar_path = weka_jar_path
        if not os.path.exists(self.weka_jar_path):
            raise IOError(
                "Input weka.jar file, '{0}' does not exist.".format(self.weka_jar_path)
            )

        self._regex_removed = re.compile(r"_removed-\d{3}")

    @staticmethod
    def _create_final_config_file(dataset_conf_path, classifier):
        if not dataset_conf_path:
            json_config = {}
        else:
            with open(dataset_conf_path, "r") as f:
                json_config = json.load(f)

        final_config = json.dumps(
            {"model_config": classifier.dict(), "ad_config": json_config},
            indent=4,
            separators=(",", ":"),
        )
        return final_config

    @staticmethod
    def _save_model_config(config_file_path, config_data):
        with open(config_file_path, "w") as f:
            f.write(config_data)

    def fill_queue_and_create_configs(
        self,
        queue: Queue,
        classifiers: List[ClassifierSchema],
        dataset_tuples: List[List[str]],
    ):
        for dataset_tuple, classifier in product(dataset_tuples, classifiers):
            train_path, test_path, conf_path = dataset_tuple

            if not os.path.exists(train_path):
                raise IOError("Input dataset '{0}' does not exist.".format(train_path))

            # Create log_file names
            final_config_str = self._create_final_config_file(conf_path, classifier)
            hash_md5 = hashlib.md5(final_config_str.encode()).hexdigest()

            basename = os.path.basename(train_path)
            dataset_name = basename.split("_")[:2]

            removed_arr = self._regex_removed.findall(basename)
            if removed_arr:
                removed_str = removed_arr[0]
            else:
                removed_str = ""

            predict_file_path = os.path.join(
                self.log_folder,
                "_".join(dataset_name)
                + "_"
                + classifier.name
                + "_"
                + hash_md5
                + removed_str
                + ".csv",
            )
            config_file_path = os.path.join(
                self.log_folder, classifier.name + "_" + hash_md5 + ".json"
            )

            # Prepare arguments for classifier
            run_args: List[str] = []
            run_args += ["-t", train_path]  # input dataset
            run_args += ["-T", test_path]  # input dataset
            run_args += [
                "-classifications",
                "weka.classifiers.evaluation.output.prediction.CSV -p first -file {0} -suppress".format(
                    predict_file_path
                ),
            ]

            # Add Weka filters
            str_filters = '-F "weka.filters.unsupervised.attribute.RemoveByName -E ^{}$"'.format(
                ID_NAME
            ) + ' -F "weka.filters.unsupervised.attribute.RemoveByName -E ^{}$"'.format(
                OD_VALUE_NAME
            )
            for one_filter in classifier.filters:
                str_filters += '-F "{0} {1}"'.format(
                    one_filter.name, " ".join(one_filter.args)
                )
            run_args += ["-F", "weka.filters.MultiFilter {0}".format(str_filters)]

            run_args += ["-S", "1"]  # Seed
            run_args += ["-W", classifier.class_name]
            if classifier.args:
                run_args += ["--"]
                run_args += classifier.args

            run_args = [
                "java",
                "-Xmx1024m",
                "-cp",
                self.weka_jar_path,
                "weka.classifiers.meta.FilteredClassifier",
            ] + run_args

            queue.put(run_args)
            self._save_model_config(config_file_path, final_config_str)
