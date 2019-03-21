from sklearn.neighbors import KernelDensity
import numpy as np


class CLMetric:
    def findLikelihood(self, df, classes):
        df_without_class = df.iloc[:, :-1]
        unique_classes = np.unique(classes)
        num_instances = len(df_without_class)
        likelihood = [1] * num_instances

        for x in unique_classes:
            class_df = df_without_class.loc[classes == x]

            counter = 0
            for attr in class_df:
                vals = class_df[attr]
                if str(class_df.dtypes[counter]) == "float64":
                    kde = KernelDensity(bandwidth=1.0, kernel="gaussian")
                    kde.fit(vals[:, None])
                    probs = np.exp(kde.score_samples(vals[:, None]))
                    for index, prob in zip(
                        [index for index, _ in vals.iteritems()], probs
                    ):
                        likelihood[index] *= prob
                else:
                    counts = dict(zip(*np.unique(vals, return_counts=True)))
                    length = len(vals)
                    for index, row in vals.iteritems():
                        likelihood[index] *= counts[row] / length
                counter += 1
        return likelihood
