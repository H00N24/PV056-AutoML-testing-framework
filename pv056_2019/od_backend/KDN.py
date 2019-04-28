import numpy as np

from sklearn.neighbors import NearestNeighbors

# TODO add prunning, make it faster


class KDNMetric:
    def countKDN(self, df, classes, k):

        values = np.empty([0, 0])

        estimator = NearestNeighbors(n_neighbors=k)
        estimator.fit(df.values)
        _, indices = estimator.kneighbors()
        for index, neighbors in enumerate(indices):
            value = 0.0
            for neighbor in neighbors:
                if classes[index] != classes[neighbor]:
                    value += 1.0
            values = np.append(values, np.full((1, 1), value / k))
        return values
