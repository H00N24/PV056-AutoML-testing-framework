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
class F3(AbstractDetector):
    name = "F3"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        classes = list(set(bin_dataframe.iloc[:, -1].values))
        num_rows = bin_dataframe.shape[0]
        max_ratios_arr = []
        df1 = bin_dataframe.loc[bin_dataframe.iloc[:, -1] == classes[0]]
        df2 = bin_dataframe.loc[bin_dataframe.iloc[:, -1] == classes[1]]
        df1.drop(df1.columns[len(df1.columns) - 1], axis=1, inplace=True)
        df2.drop(df2.columns[len(df2.columns) - 1], axis=1, inplace=True)

        for col in df1:
            overlap_min = max(df1[col].min(), df2[col].min())
            overlap_max = min(df1[col].max(), df2[col].max())
            num_overlaps = len(
                bin_dataframe[
                    (bin_dataframe[col] <= overlap_max)
                    & (bin_dataframe[col] >= overlap_min)
                ]
            )
            ratio = (num_rows - num_overlaps) / num_rows
            max_ratios_arr.append(ratio)

        self.values = np.array([max(max_ratios_arr)] * bin_dataframe.shape[0])
        return self


@detector
class F4(AbstractDetector):
    name = "F4"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        classes = list(set(bin_dataframe.iloc[:, -1].values))
        num_rows_initial = bin_dataframe.shape[0]

        while len(bin_dataframe.columns) != 1:  # only class left
            num_rows = bin_dataframe.shape[0]
            df1 = bin_dataframe.loc[bin_dataframe.iloc[:, -1] == classes[0]]
            df2 = bin_dataframe.loc[bin_dataframe.iloc[:, -1] == classes[1]]
            df1.drop(df1.columns[len(df1.columns) - 1], axis=1, inplace=True)
            df2.drop(df2.columns[len(df2.columns) - 1], axis=1, inplace=True)

            feature, overlap_min, overlap_max = self.find_best_F3_feature(
                bin_dataframe, df1, df2, num_rows
            )
            bin_dataframe = bin_dataframe.loc[
                (bin_dataframe[feature] >= overlap_min)
                & (bin_dataframe[feature] <= overlap_max)
            ]
            bin_dataframe.drop(columns=[feature], inplace=True)

        rows_left = bin_dataframe.shape[0]
        F4_measure = (num_rows_initial - rows_left) / num_rows_initial
        self.values = np.array([F4_measure] * bin_dataframe.shape[0])
        return self

    @staticmethod
    def find_best_F3_feature(bin_dataframe, df1, df2, num_rows):
        feature = None
        max_ratio = 0
        feature_overlap_min = 0
        feature_overlap_max = 0

        for col in df1:
            overlap_min = max(df1[col].min(), df2[col].min())
            overlap_max = min(df1[col].max(), df2[col].max())
            num_overlaps = len(
                bin_dataframe[
                    (bin_dataframe[col] <= overlap_max)
                    & (bin_dataframe[col] >= overlap_min)
                ]
            )
            ratio = (num_rows - num_overlaps) / num_rows
            if ratio > max_ratio:
                feature = col
                max_ratio = ratio
                feature_overlap_min = overlap_min
                feature_overlap_max = overlap_max

        return feature, feature_overlap_min, feature_overlap_max
