import numpy as np

from sklearn.tree import DecisionTreeClassifier

# TODO add prunning, make it faster


class DSMetric:
    def countDS(self, df, classes):

        values = np.empty([0, 0])

        estimator = DecisionTreeClassifier()
        estimator.fit(df, classes)
        leafsIndexes = estimator.apply(df, check_input=True)
        leafs = np.zeros(estimator.tree_.node_count)
        # count number of instances for every leaf
        for leafIndex in leafsIndexes:
            leafs[leafIndex] += 1

        biggestDisjunct = max(leafs) - 1
        # count fraction for every instance
        for leafIndex in leafsIndexes:
            values = np.append(
                values, np.full((1, 1), ((leafs[leafIndex] - 1) / biggestDisjunct) * -1)
            )

        return values
