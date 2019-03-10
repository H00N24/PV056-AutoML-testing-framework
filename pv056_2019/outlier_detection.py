from __future__ import absolute_import

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


DETECTORS: Dict[str, Any] = {}


class AbstractDetector:
    name: str
    data_type: str
    values: np.array

    def __init__(self, **settings):
        self.settings = settings

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        raise NotImplementedError()


def detector(cls):
    DETECTORS.update({cls.name: cls})
    return cls


@detector
class LOF(AbstractDetector):
    name = "LOF"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = LocalOutlierFactor(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf._decision_function(bin_dataframe.values)
        return self


@detector
class NN(AbstractDetector):
    name = "NearestNeighbors"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        if "n_neighbors" in self.settings:
            self.settings["n_neighbors"] = int(self.settings["n_neighbors"])
        self.clf = NearestNeighbors(**self.settings)
        self.clf.fit(bin_dataframe.values)
        distances, _ = self.clf.kneighbors()
        self.values = np.mean(distances, axis=1)
        return self


@detector
class IsoForest(AbstractDetector):
    name = "IsolationForest"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = IsolationForest(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf.decision_function(bin_dataframe.values)
        return self


@detector
class F2(AbstractDetector):
    name = "F2"
    data_type = "REAL"
    values = []

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        class_column = dataframe[dataframe.columns[-1]]
        classes_list = class_column.unique()

        class1 = dataframe[dataframe[class_column.name] == classes_list[0]]
        class2 = dataframe[dataframe[class_column.name] == classes_list[1]]

        bin_dataframe = dataframe._binarize_categorical_values()
        for index in range(len(bin_dataframe.columns)):
            print(len(bin_dataframe.columns))
            class1_column = class1[class1.columns[index]]
            class2_column = class2[class2.columns[index]]
            f2_result = (
                    min(class1_column.max(), class2_column.max()) -
                    max(class1_column.min(), class2_column.min()) /
                    max(class1_column.max(), class2_column.max()) -
                    min(class1_column.min(), class2_column.min()))
            self.values.append(f2_result)

        return self
