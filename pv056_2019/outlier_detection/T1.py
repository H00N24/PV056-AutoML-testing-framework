import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hcluster


class T1Metric:
    @staticmethod
    def compute_values(df: pd.DataFrame):
        thresh = 1.5
        clusters = hcluster.fclusterdata(df, thresh, criterion="distance")
        unique_clusters = np.unique(clusters, return_counts=True)
        number_of_clusters_with_one_element = (unique_clusters[0] == 1).sum()
        samples_count = len(df.index)
        return number_of_clusters_with_one_element / samples_count
