from __future__ import absolute_import

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

DETECTORS = []


class AbstractDetector:
    name: str
    data_type: str
    values: np.array

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        raise NotImplementedError()


def detector(cls):
    DETECTORS.append(cls)
    return cls


@detector
class LOF(AbstractDetector):
    name = "LOF"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        self.clf = LocalOutlierFactor(contamination="auto")
        self.clf.fit(dataframe.values)
        self.values = self.clf._decision_function(dataframe.values)
        return self


@detector
class NN(AbstractDetector):
    name = "NearestNeighbors"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        self.clf = NearestNeighbors(n_neighbors=20)
        self.clf.fit(dataframe.values)
        distances, _ = self.clf.kneighbors()
        self.values = np.mean(distances, axis=1)
        return self
