# from scipy.stats import gaussian_kde
import numpy as np


class CLDMetric:
    def findLikelihood(self, df, classes):
        df_without_class = df.iloc[:, :-1]
        unique_classes = np.unique(classes)
        num_instances = len(df_without_class)
        likelihood = [1] * num_instances
        probs_cal = {}
        for inst_class in unique_classes:
            class_df = df_without_class.loc[classes == inst_class]

            counter = 0
            class_probs_cal = {}
            for cl in class_df:
                if str(class_df.dtypes[counter]) == "float64":
                    # gs = gaussian_kde(np.array(class_df[cl]))
                    # class_probs_cal[cl] = (True, counts, length)
                    pass
                else:
                    counts = dict(zip(*np.unique(class_df[cl], return_counts=True)))
                    length = len(class_df[cl])
                    for index, row in class_df[cl].iteritems():
                        likelihood[index] *= counts[row] / length
                    class_probs_cal[cl] = (False, counts, length)
                counter += 1
            probs_cal[inst_class] = class_probs_cal

        for idx, inst_class in enumerate(unique_classes):
            class_df = df_without_class.loc[classes == inst_class]
            complement = np.delete(unique_classes, idx)
            likelihood_diff = []
            for complement_class in complement:
                likelihood_diff_item = [1] * num_instances
                for idx, val in probs_cal[complement_class].items():
                    if probs_cal[complement_class][cl][0]:
                        pass
                    else:
                        counts = probs_cal[complement_class][cl][1]
                        length = probs_cal[complement_class][cl][2]
                        for index, row in class_df[cl].iteritems():
                            try:
                                likelihood_diff_item[index] *= counts[row] / length
                            except KeyError:
                                likelihood_diff_item[index] = 0
                likelihood_diff.append(likelihood_diff_item)
            likelihood_diff = np.max(likelihood_diff, axis=0)
            for cl in class_df:
                for index, _ in class_df[cl].iteritems():
                    likelihood[index] -= likelihood_diff[index]

        return likelihood
