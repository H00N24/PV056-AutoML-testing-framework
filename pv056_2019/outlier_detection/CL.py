from sklearn.neighbors import KernelDensity
import numpy as np


class CLMetric:
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

        # Likelihood of the instance belonging to the class
        likelihood = [1] * num_instances

        # Iterates unique classes
        for x in unique_classes:
            class_df = df_without_class.loc[classes == x]
            counter = 0

            # For each attribute of the class
            for attr in class_df:

                # Values for the given attribute
                vals = class_df[attr]

                # If the attribute is a float it is a continuous variable
                # so Kernel Density is used to determine probability
                # if not isinstance(attributes_dict[attr],(list,)) and attributes_dict[attr].lower() in {"numeric", "real", "integer"}:

                if attr[1].lower() in {"numeric", "real", "integer"}:

                    kde = KernelDensity(**self.params)
                    kde.fit(vals[:, None])
                    probs = np.exp(kde.score_samples(vals[:, None]))

                    for index, prob in zip(
                        [index for index, _ in vals.iteritems()], probs
                    ):
                        likelihood[index] *= prob

                # Otherwise we count the number of occurences
                else:
                    counts = dict(zip(*np.unique(vals, return_counts=True)))
                    length = len(vals)
                    for index, row in vals.iteritems():
                        likelihood[index] *= counts[row] / length
                counter += 1
        # return likelihood
        return 1 - np.array(likelihood)
