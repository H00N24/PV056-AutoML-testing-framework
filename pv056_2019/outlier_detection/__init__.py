from __future__ import absolute_import

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import IsolationForest
from pv056_2019.outlier_detection.CL import CLMetric
from pv056_2019.outlier_detection.CLD import CLDMetric
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# from sklearn.neighbors import KNeighborsClassifier
from pv056_2019.outlier_detection.F2 import F2Metric
from pv056_2019.outlier_detection.T1 import T1Metric
from pv056_2019.outlier_detection.MV import MVMetric
from pv056_2019.outlier_detection.CB import CBMetric
from pv056_2019.outlier_detection.TD import TDMetric
from pv056_2019.outlier_detection.DCP import DCPMetric
from pv056_2019.outlier_detection.DS import DSMetric
from pv056_2019.outlier_detection.KDN import KDNMetric
from pv056_2019.outlier_detection.CODB import CODBMetric


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
# k-Disagreeing neighbors: The percentage of the
# k nearest neighbors (using Euclidean
# distance) for an instance that do not share its target class value
# TODO possibility of adding k into config file
class KDN(AbstractDetector):
    name = "KDN"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        if "n_neighbors" in self.settings:
            k = int(self.settings["n_neighbors"])
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = KDNMetric()
        self.values = self.clf.countKDN(bin_dataframe, classes, k)
        # print("KDN done sucessfully!")
        return self


@detector
# Disjunct size: The number of instances covered by a disjunct that the investigated instance
# belongs to divided by the number of instances covered by the largest disjunct in an
# unpruned decision tree
class DS(AbstractDetector):
    name = "DS"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = DSMetric()
        self.values = self.clf.countDS(bin_dataframe, classes)
        # print("DS done sucessfully!")
        return self


@detector
# Disjunct class percentage: The number of instances in a disjunct that have the same class
# label as the investigated instance divided by the total number of instances in the disjunct in a
# pruned decision tree
class DCP(AbstractDetector):
    name = "DCP"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        min_impurity_split = float(self.settings.get("min_impurity_split", 0.5))
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = DCPMetric()
        self.values = self.clf.countDCP(bin_dataframe, classes, min_impurity_split)
        # print("DCP done sucessfully!")
        return self


@detector
# Tree depth: The depth of the leaf node that classifies an instance in an induced decision tree without prunning
class TD(AbstractDetector):
    name = "TD"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = TDMetric()
        self.values = self.clf.findLeafDepthWithoutPrunning(bin_dataframe, classes)
        # print("TD without prunning done sucessfully!")
        return self


@detector
# Tree depth: The depth of the leaf node that classifies an instance in an induced decision tree with prunning
class TDWithPrunning(AbstractDetector):
    name = "TDWithPrunning"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        min_impurity_split = float(self.settings.get("min_impurity_split", 0.5))
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = TDMetric()
        self.values = self.clf.findLeafDepthWithPrunning(
            bin_dataframe, classes, min_impurity_split
        )
        # print("TD with prunning done sucessfully!")
        return self


# @detector
# # Error rate of 1NN classifier: Leave-one-out error estimate of 1NN
# class N3(AbstractDetector):
#     name = "N3"
#     data_type = "REAL"

#     def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
#         self.sum = 0
#         self.clf = NearestNeighbors(n_neighbors=1)
#         classColumnIndex = len(dataframe.columns) - 1
#         classColumnName = dataframe.columns[classColumnIndex]
#         neigh = KNeighborsClassifier(n_neighbors=1)
#         bin_dataframe = dataframe._binarize_categorical_values()
#         for index, row in dataframe.iterrows():
#             # print("Training " + repr(index) + ". classifier.")
#             leaveOne = dataframe.index.isin([index])
#             dataframeMinusOne = (dataframe[~leaveOne]).reset_index(drop=True)
#             bin_dataframeMinusOne = bin_dataframe[~leaveOne].reset_index(drop=True)
#             neigh.fit(bin_dataframeMinusOne, dataframeMinusOne[classColumnName])
#             indices = neigh.kneighbors(
#                 bin_dataframe[leaveOne], n_neighbors=1, return_distance=False
#             )
#             for i in indices:
#                 if (
#                     dataframeMinusOne[classColumnName][i[0]]
#                     != dataframe[classColumnName][index]
#                 ):
#                     self.sum += 1
#         self.values = self.sum / len(dataframe)
#         # print("N3 done sucessfully!")
#         return self


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
class CL(AbstractDetector):
    name = "ClassLikelihood"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = CLMetric(self.settings)
        self.values = self.clf.findLikelihood(bin_dataframe, classes)
        return self


@detector
class CLD(AbstractDetector):
    name = "ClassLikelihoodDifference"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = CLDMetric(self.settings)
        self.values = self.clf.findLikelihood(bin_dataframe, classes)
        return self


@detector
class F2(AbstractDetector):
    name = "F2"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = F2Metric()
        self.values = self.clf.compute_values(df=bin_dataframe, classes=classes)
        return self


@detector
class T1(AbstractDetector):
    name = "T1"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        self.clf = T1Metric()
        self.values = self.clf.compute_values(df=bin_dataframe)
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


@detector
class T2(AbstractDetector):
    name = "T2"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        samples_count = len(bin_dataframe.index)
        features_count = len(bin_dataframe.columns)
        self.values = samples_count / features_count
        return self


"""
@detector
class N1(AbstractDetector):
    name = "N1"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = N1Metric(self.settings)
        self.values = self.clf.findFraction(bin_dataframe, classes)
        return self

@detector
class N2(AbstractDetector):
    name = "N2"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = N2Metric(self.settings)
        self.values = self.clf.findFraction(bin_dataframe, classes)
        return self
"""


@detector
class MV(AbstractDetector):
    name = "MV"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = MVMetric()
        self.values = self.clf.compute_values(classes=classes)
        return self


@detector
class CB(AbstractDetector):
    name = "CB"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = CBMetric()
        self.values = self.clf.compute_values(classes=classes)
        return self


@detector
class CODB(AbstractDetector):
    name = "CODB"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = CODBMetric(self.settings)
        self.values = self.clf.compute_values(df=dataframe, classes=classes)
        return self
