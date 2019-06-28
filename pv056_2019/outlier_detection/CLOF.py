import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


class CLOFMetric:

    def __init__(self, **kwargs):
        self.alfa = float(kwargs["alfa"]) if ("alfa" in kwargs) else 0.5
        self.beta = float(kwargs["beta"]) if ("beta" in kwargs) else 0.5

    def compute_values(self, dataframe, classes):
        bin_dataframe = dataframe._binarize_categorical_values()
        unique_classes = np.unique(classes)
        class_separated_dataframes = [
            bin_dataframe.loc[classes == cl] for cl in unique_classes
        ]

        clf = LocalOutlierFactor()
        clf.fit(bin_dataframe.values)
        df_lof = clf._decision_function(bin_dataframe.values)

        class_separated_lof = []
        for df in class_separated_dataframes:
            class_separated_lof.append(clf._decision_function(df.values))

        values = [0] * len(dataframe)

        for index, row in dataframe.iterrows():
            class_column = dataframe.columns.values[-1]
            row_class_index = np.where(unique_classes == row[class_column])[0][0]

            other_classes_dataframes_indexes = list(range(len(class_separated_dataframes)))
            other_classes_dataframes_indexes.remove(row_class_index)
            other_classes_dataframe = pd.DataFrame()
            other_classes_dataframe = other_classes_dataframe.append(
                [class_separated_dataframes[i] for i in other_classes_dataframes_indexes]
            )
            other_classes_dataframe = other_classes_dataframe.append(bin_dataframe.loc[index])
            other_classes_lof = clf._decision_function(other_classes_dataframe.values)

            row_location = class_separated_dataframes[row_class_index].index.get_loc(index)
            same_lof = class_separated_lof[row_class_index][row_location]
            other_lof = other_classes_lof[-1]
            all_lof = df_lof[index]

            values[index] = same_lof + self.alfa * (1 / other_lof) + self.beta * all_lof

        return values
