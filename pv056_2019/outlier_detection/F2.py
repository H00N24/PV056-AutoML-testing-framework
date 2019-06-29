import numpy as np
import pandas as pd

from collections import namedtuple

Extremes = namedtuple("Extremes", "min max")


class F2Metric:
    @staticmethod
    def compute_values(df, classes):
        classes_list = np.unique(classes)
        class_instance_indexes = [df.loc[classes == cl].index for cl in classes_list]
        class_instances = [
            pd.concat([df.loc[[index]] for index in instance])
            for instance in class_instance_indexes
        ]
        result = 1
        for ft in range(len(df.columns)):

            class_feature_values = [
                class_instances[cl].loc[:, class_instances[cl].columns[ft]]
                for cl in range(len(classes_list))
            ]
            class_extremes = [
                Extremes(class_feature_values[cl].min(), class_feature_values[cl].max())
                for cl in range(len(classes_list))
            ]

            result *= F2Metric._f2_step(class_extremes=class_extremes)

        return result

    @staticmethod
    def _f2_step(class_extremes):
        return (
            min(class_extremes[0].max, class_extremes[1].max)
            - max(class_extremes[0].min, class_extremes[1].min)
        ) / (
            max(class_extremes[0].max, class_extremes[1].max)
            - min(class_extremes[0].min, class_extremes[1].min)
        )
