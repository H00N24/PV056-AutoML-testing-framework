from datetime import datetime
import subprocess
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

        # Prepare arguments for classifier
        clf_args['-t'] = dataset_path   # training dataset
        clf_args['-x'] = '5'            # cross-validation folds
        clf_args['-classifications'] = 'weka.classifiers.evaluation.output.prediction.CSV -p 1-last -file {0} -suppress'.\
            format(output_file)         # output in CSV format

        # Run classifier
        print("-" * 40)
        print("Running \'{0}\' with:".format(clf_name))
        print('\n'.join('\t{0:18} {1}'.format(key, value) for key, value in clf_args.items()))

        # Convert args to args_list
        args_list = ['java', '-Xmx1024m', '-cp', self.weka_jar_path, clf_name] 
        for key, value in clf_args.items():
            args_list += [key, value]

        # Run weka
        p = subprocess.Popen(args_list,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        output, err = p.communicate()
        rc = p.returncode

        ## Run J48 classifier
        # java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff -x 5 -classifications "weka.classifiers.evaluation.output.prediction.CSV -p 1-last"
        # java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff -x 5 -classifications "weka.classifiers.evaluation.output.prediction.CSV -p 1-last -file output.txt -suppress"

if __name__ == "__main__":
    man = WekaManager('logs/', 'weka-3-8-3/weka.jar')
    man.run_weka_classifier('weka.classifiers.trees.J48', dict(), "weka-3-8-3/data/diabetes.arff")
