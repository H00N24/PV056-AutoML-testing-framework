import numpy as np


class MVMetric:
    @staticmethod
    def compute_values(classes: np.array):
        classes_list, counts = np.unique(classes, return_counts=True)
        majority_class_count = counts.max()
        class_sizes = dict(zip(classes_list, counts))
        values = np.empty([0])

        for cl in classes:
            values = np.append(values, class_sizes[cl] / majority_class_count)

        return values
