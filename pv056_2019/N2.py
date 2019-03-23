from sklearn.neighbors import NearestNeighbors

import numpy as np


class N2Metric:
    def findFraction(self, df, classes):
        unique_classes = np.unique(classes)
        num_instances = len(df)

        class_instances = np.array([df.loc[classes == x].index for x in unique_classes])

        self.clf = NearestNeighbors(n_neighbors=num_instances, algorithm="ball_tree")
        self.clf.fit(df.values)

        intra_class = []
        inter_class = []

        for cl in range(len(class_instances)):
            for inst in class_instances[cl]:
                n_neighbors = min(200, num_instances)

                intra_class_dist = None
                inter_class_dist = None

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
