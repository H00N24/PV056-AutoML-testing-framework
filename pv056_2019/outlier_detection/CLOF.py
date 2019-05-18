import numpy as np
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

        df_counters = [-1 for _ in class_separated_dataframes]
        values = [0] * len(dataframe)

        for index, row in dataframe.iterrows():

            class_column = dataframe.columns.values[-1]
            row_class_index = np.where(unique_classes == row[class_column])[0][0]

            df_counters[row_class_index] += 1

            other_class_lof_cumulated = 0
            for i in range(len(class_separated_dataframes)):

                if i == row_class_index:
                    continue

                class_separated_dataframes[i].loc[
                    class_separated_dataframes[i].index[-1] + 1
                ] = bin_dataframe.loc[index]

                clf.fit(class_separated_dataframes[i].values)
                joined_df_lof = clf._decision_function(class_separated_dataframes[i].values)

                class_separated_dataframes[i].drop(
                    class_separated_dataframes[i].index[-1], inplace=True
                )

                other_class_lof_cumulated += joined_df_lof[-1]

            row_class_value = class_separated_lof[row_class_index][df_counters[row_class_index]]
            other_class_value = other_class_lof_cumulated / (len(class_separated_dataframes) - 1)
            df_value = df_lof[index]

            values[index] = row_class_value + self.alfa * (1 / other_class_value) + 0.5 * df_value

        return values
