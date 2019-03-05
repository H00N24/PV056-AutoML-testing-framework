import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

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
