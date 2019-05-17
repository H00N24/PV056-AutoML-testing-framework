import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class CLOFMetric:
    @staticmethod
    def compute_values(dataframe, classes):
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

            row_class_index = np.where(unique_classes == row["Class"])[0][0]
            other_class_index = row_class_index % 1

            df_counters[row_class_index] += 1
            class_separated_dataframes[other_class_index].loc[
                class_separated_dataframes[other_class_index].index[-1] + 1
                ] = row.values[:-1]

            clf.fit(class_separated_dataframes[other_class_index].values)
            joined_df_lof = clf._decision_function(class_separated_dataframes[other_class_index].values)

            class_separated_dataframes[other_class_index].drop(
                class_separated_dataframes[other_class_index].index[-1], inplace=True
            )

            row_class_value = class_separated_lof[row_class_index][df_counters[row_class_index]]
            other_class_value = joined_df_lof[-1]
            df_value = df_lof[index]

            values[index] = row_class_value + 0.5 * (1 / other_class_value) + 0.5 * df_value

        return values
