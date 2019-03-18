# from scipy.stats import gaussian_kde
import numpy as np


class CLDMetric:
    def findLikelihood(self, df, classes):
        df_without_class = df.iloc[:, :-1]
        unique_classes = np.unique(classes)
        num_instances = len(df_without_class)
        likelihood = [1] * num_instances
        probs_cal = []
        for x in unique_classes:
            class_df = df_without_class.loc[classes == x]

            counter = 0
            class_probs_cal = []
            for cl in class_df:
                if str(class_df.dtypes[counter]) == "float64":
                    # gs = gaussian_kde(np.array(class_df[cl]))
                    pass
                else:
                    counts = dict(zip(*np.unique(class_df[cl], return_counts=True)))
                    length = len(class_df[cl])
                    for index, row in class_df[cl].iteritems():
                        likelihood[index] *= counts[row] / length
                    class_probs_cal.append((counts, length))
                counter += 1
            probs_cal.append(class_probs_cal)

        for x in unique_classes:

            print(class_df)
            exit()

            """counter = 0
            class_probs_cal = []
            for cl in class_df:
                if str(class_df.dtypes[counter]) == "float64":
                    gs = gaussian_kde(np.array(class_df[cl]))
                else:
                    counts = dict(zip(*np.unique(class_df[cl], return_counts=True)))
                    length = len(class_df[cl])
                    for index, row in class_df[cl].iteritems():
                        likelihood[index] *= counts[row] / length
                    class_probs_cal.append((counts, length))
                counter += 1
            probs_cal.append(class_probs_cal)"""
        return likelihood
