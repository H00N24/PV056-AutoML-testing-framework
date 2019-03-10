from sklearn.metrics import pairwise_distances

# from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import numpy as np


class N1Metric:
    def findFraction(self, df):
        distances = np.triu(pairwise_distances(df.values[:, :-1]))
        mst = minimum_spanning_tree(distances)
        vertices = [0] * len(df)

        for x in np.dstack(mst.nonzero())[0]:
            if df.iloc[:, -1][x[0]] != df.iloc[:, -1][x[1]]:
                vertices[x[0]] = 1
                vertices[x[1]] = 1
        return sum(vertices) / len(vertices)


if __name__ == "__main__":
    import sys

    df = pd.read_csv(sys.argv[1], header=None)
    clf = N1Metric()
    print(clf.findFraction(df))
