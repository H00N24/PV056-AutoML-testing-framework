from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np


class N1Metric:
    def findFraction(self, df, classes):
        distances = np.triu(pairwise_distances(df.values))
        mst = minimum_spanning_tree(distances)
        vertices = [0] * len(df)

        for x in np.dstack(mst.nonzero())[0]:
            if classes[x[0]] != classes[x[1]]:
                vertices[x[0]] = 1
                vertices[x[1]] = 1

        return sum(vertices) / len(vertices)


# if __name__ == "__main__":
#    import sys
#
#    df = pd.read_csv(sys.argv[1], header=None)
#    clf = N1Metric()
#    print(clf.findFraction(df))
