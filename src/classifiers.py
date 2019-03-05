from datetime import datetime
import subprocess
import copy
import os


class WekaManager():

    # Weka classifiers
    # http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html

    def __init__(self, log_folder, weka_jar_path):
        self.log_folder = log_folder
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)
        self.weka_jar_path = weka_jar_path
        if not os.path.exists(self.weka_jar_path):
            raise IOError("Input weka.jar file, \'{0}\' does not exist.".
                format(self.weka_jar_path))

    def run_weka_classifier(self, clf_name, clf_args, dataset_path):
        # Check dataset path & classifier name
        if not os.path.exists(dataset_path):
            raise IOError("Input dataset \'{0}\' does not exist.".
                format(dataset_path))

        # Prepare output file
        datetime_now_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.log_folder,
                                   clf_name.split('.')[-1] + "_" + datetime_now_str + '.csv')

        # Check clf_args
        if '-t' in clf_args or '-x' in clf_args or '-classifications' in clf_args:
            print("Settings \'-t\', \'-x\', \'-classifications\' will be overwritten.")

        # Prepare arguments for classifier
        run_args = copy.copy(clf_args)
        run_args['-t'] = dataset_path   # training dataset
        run_args['-x'] = '5'            # cross-validation folds
        run_args['-classifications'] = 'weka.classifiers.evaluation.output.prediction.CSV -p 1-last -file {0} -suppress'.\
            format(output_file)         # output in CSV format

        # Run classifier
        print("-" * 40)
        print("Running \'{0}\' with:".format(clf_name))
        print('\n'.join('\t{0:18} {1}'.format(key, value) for key, value in run_args.items()))

        # Convert args to args_list
        args_list = ['java', '-Xmx1024m', '-cp', self.weka_jar_path, clf_name] 
        for key, value in run_args.items():
            args_list += [key, value]

        # Run weka
        p = subprocess.Popen(args_list,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode

        if rc != 0:
            print("[ERROR] Something went wrong!")


def load_config(config_path):
    import json
    with open(config_path, 'r') as conf_f:
        configuration = json.load(conf_f)

    try:
        weka_jar_path = configuration['weka_jar_path']
        classifiers = configuration['classifiers']
        datasets = configuration['datasets']
    except KeyError as err:
        raise KeyError("Parameter {0} is missing in config file.".format(err))

    return weka_jar_path, classifiers, datasets

def _yield_dataset_path(datasets):
    if 'root_folder' not in datasets:
        raise KeyError("Parameter \'{0}\' is missing in config file.".format('root_folder'))
    if 'names' not in datasets:
        raise KeyError("Parameter \'{0}\' is missing in config file.".format('names'))

    for file_name in datasets['names']:
        yield os.path.join(datasets['root_folder'], file_name)

def _yield_classificier(classifiers):
    for clf in classifiers:
        if 'class_name' not in clf:
            raise KeyError("Parameter \'{0}\' is missing in config file.".format('class_name'))
        clf_name = clf['class_name']
        clf_args = clf['args'] if 'args' in clf else dict()
        yield clf_name, clf_args

def _valid_config_path(path):
    import argparse
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Invalid path to config file.")
    else:
        return path

def main_run():
    import argparse
    parser = argparse.ArgumentParser(description="PV056-AutoML-testing-framework")
    parser.add_argument('-c', '--config',
                        type=_valid_config_path,
                        help='path to config file',
                        required=True)
    args = parser.parse_args()

    weka_jar, classfiers, datasets = load_config(args.config)
    weka_man = WekaManager('logs/', weka_jar)

    for clf_name, clf_args in _yield_classificier(classfiers):
        for dataset_path in _yield_dataset_path(datasets):
            weka_man.run_weka_classifier(clf_name,
                                         clf_args,
                                         dataset_path)

if __name__ == "__main__":
    main_run()
