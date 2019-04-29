from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np


class N1Metric:
    def __init__(self, params):
        self.params = params

    def findFraction(self, df, classes):

        # Upper triangle of distance matrix
        distances = np.triu(pairwise_distances(df.values, **self.params))

        # Finds minimum spanning tree
        mst = minimum_spanning_tree(distances)

        # Vertices that are connected to the different class
        vertices = [0] * len(df)

        for x in np.dstack(mst.nonzero())[0]:
            if classes[x[0]] != classes[x[1]]:
                vertices[x[0]] = 1
                vertices[x[1]] = 1

        return sum(vertices) / len(vertices)
