from sklearn.metrics import pairwise_distances
import numpy as np


class N2Metric:
    def findFraction(self, df, classes):
        unique_classes = np.unique(classes)
        # class_instances = [df.loc[classes == x].values for x in unique_classes]
        class_instances = [df.loc[classes == x].index for x in unique_classes]
        distances = pairwise_distances(df.values)
        intra_class = []
        inter_class = []
        for cl in range(len(class_instances)):
            for inst in range(len(class_instances[cl])):
                # within class
                intra_class.append(
                    np.min(
                        np.take(
                            distances[class_instances[cl][inst]],
                            np.delete(class_instances[cl], inst),
                        )
                    )
                )
                inter_class.append(
                    np.min(
                        np.take(
                            distances[class_instances[cl][inst]],
                            np.concatenate(np.delete(class_instances, cl)),
                        )
                    )
                )

        return np.mean(intra_class) / np.mean(inter_class)


# if __name__ == "__main__":
#    import sys
#
#    df = pd.read_csv(sys.argv[1], header=None)
#    clf = N2Metric()
#    print(clf.findFraction(df))
