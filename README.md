# PV056-AutoML-testing-framework
* *PV056 Machine learning and knowledge discovery*

## How to use PV056-AutoML-testing-framework
* First, follow the [Installation guide](#installation-guide) section
* Then follow the [Usage](#usage) section
 
## Installation guide 
### Prerequisites
- Python version >=3.6 (including python3-dev, python3-wheel)
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/)

### Installation to python virtual env (recommended)
It's highly recommended to install this testing framework to python virtual environment.
- Simple python virtual environment guide: https://realpython.com/python-virtual-environments-a-primer/

Simply run commands below in the root folder of this repository.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -e .
```

### Installation without python virtual env (not recommended)
```
$ pip install .
```


### Downloading datasets
All data files are from [OpenML](https://www.openml.org).

Data files are compressed in `data/openML-datasets.zip` (you have to unzip it). Because this file is larger than 50mb we are using git lfs (large file storage). You can read the documentation [here](https://git-lfs.github.com).

#### TL;DR
Run commands below in the root folder of this repo.
```
$ sudo apt install git-lfs
$ git lfs install
$ git lfs pull
$ cd data && unzip openML-datasets.zip
```


## Usage
If you have chosen to install this tester in the virtual environment, you must activate it to proceed.
* *Pipeline:* enrich data -> run classifiers -> (optional) statistics

### Enriching data with outlier detection methods
This step enriches datasets with new values from all outlier methods specified in a configuration file and adds a new column "INDEX" which we need for weka. This column is not used for classification. It also generates a configuration file (`datasets-config-file`), which contains paths to the new datasets for the second part of this framework.

```
(venv)$ pv056-enrich-data --help

usage: pv056-enrich-data [-h] --config-file CONFIG_FILE --datasets-config-file
                         DATASETS_CONFIG_FILE

Script enriches datasets with new values based on outlier detection methods.
Available outlier detection methods: LOF, NearestNeighbors, IsolationForest

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-config-file DATASETS_CONFIG_FILE, -d DATASETS_CONFIG_FILE
                        Filename of output datasets config
```

#### Example usage
```
(venv)$ pv056-enrich-data -c config_enrich_example.json -d config_datasets_example.json
```

#### Example of configuration file
* *data_paths*
    * List of directories/filenames of datasets in arff format
* *output_dir*
    * Directory where generated datasets should be saved
* *detetors*
    * Dictionary "Outlier detector name" : {**parametrs}

```json
{
    "data_paths": [
        "data/"
    ],
    "output_dir": "enriched_datasets/",
    "detectors": {
        "LOF": {
            "contamination": "auto"
        },
        "NearestNeighbors": {
            "n_neighbors": 20
        },
        "IsolationForest": {
            "behaviour": "new",
            "contamination": "auto"
        }
    }
}
```
To run it without any outlier detection methods use the config below.
```json
{
    "data_paths": [
        "data/"
    ],
    "output_dir": "enriched_datasets/",
    "detectors": {}
}
```

#### Outlier detector names and parameters:
| Name | Full name | Parameters |
|:----:|:----------:|:----------:|
| **LOF** | Local Outlier Factor | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| **NearestNeighbors** | Nearest Neighbors | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) |
| **IsolationForest** | Isolation Forest | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
| **F2** | Max individual feature efficiency | -- |
| **T1** | Fraction of maximum covering spheres | -- |
| **T2** | Ave number of points per dimension | -- |
| **MV** | Minority value | -- |
| **CB** | Class balance | -- |

* New methods for outlier detection coming soon!

### Run weka classifiers
To run a weka classifier using this framework, first setup virtual environment, install required modules and download weka tool.
1) Activate your virtual Python environment with this project.
2) Generate `config_datasets_example.json` configuration file using `pv056-enrich-data` (See [Enrich data](#enriching-data-with-outlier-detection-methods))
3) Create a `config_clf_example.json` file, with weka classifiers and their configuration (See [Config file for weka classifiers](#example-of-config-file-for-weka-classifiers))
5) Run `pv056-run-clf` script, see command below

```
(venv)$ pv056-run-clf --help

usage: pv056-run-clf [-h] -cc CONFIG_CLF -cd CONFIG_DATA

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -cc CONFIG_CLF, --config-clf CONFIG_CLF
                        path to classifiers config file
  -cd CONFIG_DATA, --config-data CONFIG_DATA
                        path to datasets config file
```

#### Example usage
```
(venv)$ pv056-run-clf --config-clf config_clf_example.json --config-data config_datasets_example.json
```

#### Example of config file for weka classifiers
* *output_folder*
    * path to output folder, where outputs from your classifiers will be saved
* *weka_jar_path*
    * path to a weka.jar file
* *classifiers*
    * list of classifiers which you want to run
    * you can run an arbitrary number of classifiers, even same classifier with different configuration
    * list of [weka classifiers](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html)
    * *class_name*
        * name of weka classifier class
    * *args*
        * optional value
        * list of arguments for specific classifier
        * you can find all arguments for specific classifier using weka command: ```$ java -cp weka.jar weka.classifiers.trees.J48 --help``` or in [weka documentation](http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html)
        * you can find more information about Weka-CLI in the section below [How to work with Weka 3](#how-to-work-with-Weka-3), but I don't think you need that for using this tool.
    * *filters*
        * optional value
        * you can use any filter from Weka
        * you have to specify name of filter and arguments for it

```json
{
    "output_folder": "clf_outputs/",
    "weka_jar_path": "weka-3-8-3/weka.jar",
    "classifiers": [
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C", 0.25,
                "-M", 2
            ],
            "filters": [
                {
                    "name": "weka.filters.unsupervised.attribute.RemoveByName",
                    "args": [
                        "-E", "^size$"
                    ]
                }
            ]
        },
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C", 0.35
            ]
        },
        {
            "class_name": "weka.classifiers.bayes.BayesNet"
        }
    ]
}

```

### Count accuracy
To count accuracy simply run `pv056-statistics` script. In the future, we will add Precision and Recall.
```
(venv)$ pv056-statistics --help
usage: pv056-statistics [-h] --results-dir RESULTS_DIR [--pattern PATTERN]

Script for counting basic statistic (Accuracy, )

optional arguments:
  -h, --help            show this help message and exit
  --results-dir RESULTS_DIR, -r RESULTS_DIR
                        Directory with results in .csv
  --pattern PATTERN, -p PATTERN
                        Regex for filename (Python regex)
```
#### Example
```
(venv)$ pv056-statistics -r clf_outputs/ -p "teaching.*"
teachingAssistant BayesNet 3e408e23621de037f4751689311eb00d.csv
         Accuracy: 0.9073
teachingAssistant J48 81498a187313e89f240c8ead4557906b.csv
         Accuracy: 0.5232
teachingAssistant J48 9f0cf2e85982a05ecf632ee428274ec3.csv
         Accuracy: 0.5166
```

## How to work with Weka 3
* Download Weka from https://www.cs.waikato.ac.nz/ml/weka/downloading.html
* Weka classifiers http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
* Documentation to Weka in `The WEKA Workbench` & `Weka manual` documents at https://www.cs.waikato.ac.nz/ml/weka/documentation.html
```shell
# Example of running J48 classifier with diabetes dataset
java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff

# General & Specific configuration options
java -cp weka.jar weka.classifiers.trees.J48 --help
```

## Developers guide

As developers, we have chosen to use [Black](https://github.com/ambv/black/) auto-formater and [Flake8](https://gitlab.com/pycqa/flake8) style checker. Both of these tools are pre-prepared for pre-commit. It's also recommended to use [mypy](https://github.com/python/mypy).


Since there is the typing module in the standard Python library, it would be a shame not to use it.  A wise old man once said: More typing, fewer bugs. [Typing module](https://docs.python.org/3/library/typing.html)


To prepare dev env run commands below.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -r requirements-dev.txt
(venv)$ pre-commit install
(venv)$ pip install -e .
```

For generating `requirements.txt` we are using pip-compile from [pip-tools](https://github.com/jazzband/pip-tools).
For keeping your packages updated, use `pip-sync requirements.txt requirements-dev.txt`.
