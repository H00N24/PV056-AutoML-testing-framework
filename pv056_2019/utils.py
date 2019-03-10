from datetime import datetime
import json


# *********************************************************
# Utils for classifiers
# *********************************************************

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def load_config_clf(config_path):
    with open(config_path, "r") as conf_f:
        config = json.load(conf_f)
    try:
        output_folder = config["output_folder"]
        weka_jar_path = config["weka_jar_path"]
        classifiers = config["classifiers"]
    except KeyError as err:
        raise KeyError("Parameter {0} is missing in config file.".format(err))

    return output_folder, weka_jar_path, classifiers


def load_config_data(config_path):
    with open(config_path, "r") as conf_f:
        config_str = json.load(conf_f)
    return config_str


def yield_classifiers(classifiers):
    for clf in classifiers:
        if "class_name" not in clf:
            raise KeyError(
                "Parameter '{0}' is missing in config file.".format("class_name")
            )
        clf_name = clf["class_name"]
        clf_args = clf["args"] if "args" in clf else list()
        yield clf_name, clf_args


def get_datetime_now_str():
    now = datetime.now()
    datetime_str = datetime.strftime(now, DATETIME_FORMAT)
    return datetime_str


def get_clf_name(clf_class):
    return clf_class.split(".")[-1]


# *********************************************************
# Other utils
# *********************************************************

"""
def calculate_dataset_hash(dataset_path, dataset_conf_path):
    with open(dataset_conf_path, 'r') as cf:
        json_str = json.dumps(json.load(cf), sort_keys=True,
                              separators=(',', ':'))
    file_name = os.path.basename(dataset_path)
    final_str = json_str + file_name
    hash_md5 = hashlib.md5(final_str.encode()).hexdigest()
    return hash_md5
"""
