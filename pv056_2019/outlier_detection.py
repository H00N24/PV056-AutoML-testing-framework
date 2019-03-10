from __future__ import absolute_import

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from N1 import N1Metric


DETECTORS: Dict[str, Any] = {}


class AbstractDetector:
    name: str
    data_type: str
    values: np.array

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

        self.clf = LocalOutlierFactor(contamination="auto")
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf._decision_function(bin_dataframe.values)
        return self


@detector
class NN(AbstractDetector):
    name = "NearestNeighbors"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = NearestNeighbors(n_neighbors=20)
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

        self.clf = IsolationForest()
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf.decision_function(bin_dataframe.values)
        return self


@detector
class N1(AbstractDetector):
    name = "N1"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = N1Metric()
        self.values = self.clf.findFraction(bin_dataframe)
        return self
