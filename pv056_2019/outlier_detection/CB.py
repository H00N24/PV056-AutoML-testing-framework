import numpy as np


class CBMetric:
    @staticmethod
    def compute_values(classes: np.array):
        classes_list, counts = np.unique(classes, return_counts=True)
        class_sizes = dict(zip(classes_list, counts))
        classes_count = len(classes_list)
        samples_count = len(classes)
        values = np.empty([0])

        for cl in classes:
            value = (class_sizes[cl] / samples_count) - (1 / classes_count)
            values = np.append(values, value)

        return values
