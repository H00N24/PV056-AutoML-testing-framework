from sklearn.neighbors import NearestNeighbors

import numpy as np


class N2Metric:
    def __init__(self, params):
        self.params = params

    def findFraction(self, df, classes):

        # List of unique classes
        unique_classes = np.unique(classes)
        num_instances = len(df)

        # Indices for instances belonging t the same class
        class_instances = np.array([df.loc[classes == x].index for x in unique_classes])

        # Nearest neighbour classifier to find the distances
        self.clf = NearestNeighbors(n_neighbors=num_instances, **self.params)
        self.clf.fit(df.values)

        intra_class = []
        inter_class = []

        # Iterates all the classes
        for cl in range(len(class_instances)):
            # Instances of that class
            for inst in class_instances[cl]:

                # Default number of neighbours in which it looks for intra
                # and inter class distances
                n_neighbors = min(200, num_instances)

                intra_class_dist = None
                inter_class_dist = None

                # It needs to find both distances otherwise the size of a
                # neighbourhood is doubled
                while inter_class_dist is None or intra_class_dist is None:
                    distances, indices = self.clf.kneighbors(
                        [df.values[inst]], n_neighbors=n_neighbors
                    )
                    for i in range(1, len(indices[0])):
                        if (
                            intra_class_dist is None
                            and indices[0][i] in class_instances[cl]
                        ):
                            intra_class_dist = distances[0][i]
                        if (
                            inter_class_dist is None
                            and indices[0][i] not in class_instances[cl]
                        ):
                            inter_class_dist = distances[0][i]
                        if inter_class_dist and intra_class_dist:
                            break
                    n_neighbors *= 2
                    if n_neighbors > num_instances:
                        n_neighbors = num_instances
                intra_class.append(intra_class_dist)
                inter_class.append(inter_class_dist)
        return np.mean(intra_class) / np.mean(inter_class)
