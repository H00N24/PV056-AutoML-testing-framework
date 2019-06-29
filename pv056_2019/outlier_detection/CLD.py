from sklearn.neighbors import KernelDensity
import numpy as np


class CLDMetric:
    def __init__(self, params):
        self.params = params

    def findLikelihood(self, df, classes):

        # Adjusts dataframe by removing the class column
        # df_without_class = df.iloc[:, :-1]
        df_without_class = df

        # Find unique classes
        unique_classes = np.unique(classes)

        # Number of all instances
        num_instances = len(df_without_class)

        # Likelihood difference of the instance belonging to the class
        likelihood = [1] * num_instances

        probs_cal = {}

        # Probability of the instance belonging to its class
        for inst_class in unique_classes:
            class_df = df_without_class.loc[classes == inst_class]

            counter = 0
            class_probs_cal = {}
            for attr in class_df:
                vals = class_df[attr]
                if attr[1].lower() in {"numeric", "real", "integer"}:

                    kde = KernelDensity(**self.params)
                    kde.fit(vals[:, None])
                    probs = np.exp(kde.score_samples(vals[:, None]))

                    for index, prob in zip(
                        [index for index, _ in vals.iteritems()], probs
                    ):
                        likelihood[index] *= prob
                    class_probs_cal[attr] = ("continuous", kde)
                else:
                    counts = dict(zip(*np.unique(vals, return_counts=True)))
                    length = len(vals)
                    for index, row in vals.iteritems():
                        likelihood[index] *= counts[row] / length
                    class_probs_cal[attr] = ("discrete", (counts, length))
                counter += 1
            probs_cal[inst_class] = class_probs_cal

        # Probability the instance belonging to the different class
        for idx, inst_class in enumerate(unique_classes):
            class_df = df_without_class.loc[classes == inst_class]
            complement = np.delete(unique_classes, idx)
            likelihood_diff = []
            for complement_class in complement:
                likelihood_diff_item = [1.0] * num_instances
                for attr, val in probs_cal[complement_class].items():
                    if val[0] == "continuous":
                        for index, row in class_df[attr].iteritems():
                            likelihood_diff_item[index] *= np.exp(val[1].score([[row]]))
                    else:
                        counts = val[1][0]
                        length = val[1][1]
                        for index, row in class_df[attr].iteritems():
                            try:
                                likelihood_diff_item[index] *= counts[row] / length
                            except KeyError:
                                likelihood_diff_item[index] = 0
                likelihood_diff.append(likelihood_diff_item)
            likelihood_diff = np.max(likelihood_diff, axis=0)
            for inst in class_df.index.values.tolist():
                likelihood[inst] -= likelihood_diff[inst]

        # return likelihood
        return 1 - np.array(likelihood)
