# PV056-AutoML-testing-framework
* *PV056 Machine learning and knowledge discovery*

## How to work with Weka 3
* Download Weka from https://www.cs.waikato.ac.nz/ml/weka/downloading.html
* Weka classifiers http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
* Documentation to Weka in `The WEKA Workbench` & `Weka manual` documents at https://www.cs.waikato.ac.nz/ml/weka/documentation.html
    * we recommend you to read the Command Line Interface sections

```shell
# Example of running J48 classifier with diabetes dataset
java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff

# General & Specific configuration options
java -cp weka.jar weka.classifiers.trees.J48 --help
```

## How to use PV056-AutoML-testing-framework
* TODO
 

## Installation guide 
### Prerequisites
- Python version >=3.6
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

Data files are compressed in `data/openML-datasets.zip`. Because this file is larger than 50mb we are using [git large file storage](https://git-lfs.github.com). 

#### TL;DR
Run commands below in the root folder of this repo.
```
$ sudo apt install git-lfs
$ git lfs install
$ git lfs pull
```


## Usage
If you have chosen to install this tester in the virtual environment, you must activate it to proceed.

### Enriching data with outlier detection methods
```
$ pv056-enrich-data --help
```
### Using weka classifiers

[//]: # (TODO)


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