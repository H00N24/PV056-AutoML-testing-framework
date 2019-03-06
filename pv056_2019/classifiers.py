from pv056_2019.utils import get_datetime_now_str, get_clf_name
import subprocess
import copy
import json
import os


class ClassifierManager:

    # Weka classifiers
    # http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html

    def __init__(self, log_folder, weka_jar_path):
        self.log_folder = log_folder
        self.weka_jar_path = weka_jar_path
        if not os.path.exists(self.weka_jar_path):
            raise IOError(
                "Input weka.jar file, '{0}' does not exist.".format(self.weka_jar_path)
            )

    @staticmethod
    def _save_stds(log_folder, output, errors):
        with open(os.path.join(log_folder, "stdout.txt"), "w") as f:
            f.write(output.decode("UTF-8"))
        with open(os.path.join(log_folder, "stderr.txt"), "w") as f:
            f.write(errors.decode("UTF-8"))

    @staticmethod
    def _save_model_config(log_folder, clf_class, clf_args):
        with open(os.path.join(log_folder, "config.json"), "w") as f:
            f.write(
                json.dumps(
                    {"class_name": clf_class, "args": clf_args},
                    indent=4,
                    separators=(",", ":"),
                )
            )

    @staticmethod
    def _print_run_info(clf_class, run_args):
        # Run classifier
        print("-" * 40)
        print("Running '{0}' with:".format(clf_class))
        for arg in run_args:
            arg_msg = " " + arg + "\n" if arg[0] != "-" else "\t" + arg
            print(arg_msg, end="")

    @staticmethod
    def run_subprocess(run_args):
        p = subprocess.Popen(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode
        return output, err, rc

    def prepare_log_folders(self, clf_class, dataset_path):
        # Create folder names
        dataset_name, _ = os.path.splitext(os.path.basename(dataset_path))
        clf_name = get_clf_name(clf_class)
        run_folder_name = get_datetime_now_str() + "_run"
        clf_folder_name = get_datetime_now_str() + "_" + clf_name + "_" + dataset_name

        # Create folders
        folder = os.path.join(self.log_folder, run_folder_name, clf_folder_name)
        os.makedirs(folder, exist_ok=True)
        return folder, clf_name

    def run_weka_classifier(self, clf_class, clf_args, dataset_path):
        # Check dataset path
        if not os.path.exists(dataset_path):
            raise IOError("Input dataset '{0}' does not exist.".format(dataset_path))

        # Prepare output folders
        log_folder, clf_name = self.prepare_log_folders(clf_class, dataset_path)

        # Check clf_args
        if "-t" in clf_args or "-x" in clf_args:
            print("Settings '-t', '-x' will be overwritten.")

        # Prepare arguments for classifier
        run_args = copy.copy(clf_args)
        run_args = [str(arg) for arg in run_args]
        run_args += ["-t", dataset_path]
        run_args += ["-x", "5"]
        # run_args += ['-d', os.path.join(log_folder, clf_name + '.model')]
        run_args += [
            "-classifications",
            "weka.classifiers.evaluation.output.prediction.CSV -p 1-last -file {0} -suppress".format(
                os.path.join(log_folder, "predict.csv")
            ),
        ]

        # Print run info
        self._print_run_info(clf_class, run_args)

        # Add some run args & run classifier
        run_args = [
            "java",
            "-Xmx1024m",
            "-cp",
            self.weka_jar_path,
            clf_class,
        ] + run_args
        output, err, rc = self.run_subprocess(run_args)

        if rc != 0:
            print("[ERROR] Something went wrong!")

        # Save weka outputs, errors and model configuration
        self._save_stds(log_folder, output, err)
        self._save_model_config(log_folder, clf_class, clf_args)

        # Predict labels
        # self.predict_labels(clf_class, clf_args, dataset_path)

    def predict_labels(self, clf_class, clf_args, dataset_path):
        clf_name = get_clf_name(clf_class)
        run_args = copy.copy(clf_args)
        run_args += ["-T", dataset_path]
        run_args += ["-l", os.path.join(self.log_folder, clf_name + ".model")]
        run_args += [
            "-classifications",
            "weka.classifiers.evaluation.output.prediction.CSV -p 1-last -file {0} -suppress".format(
                os.path.join(self.log_folder, "predict.csv")
            ),
        ]
        run_args = [
            "java",
            "-Xmx1024m",
            "-cp",
            self.weka_jar_path,
            clf_class,
        ] + run_args
        _, _, _ = self.run_subprocess(run_args)
