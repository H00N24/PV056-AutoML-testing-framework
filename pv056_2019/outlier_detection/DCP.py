import numpy as np

from sklearn.tree import DecisionTreeClassifier


class DCPMetric:
    def countDCP(self, df, classes, minimum_impurity_split):

        values = np.empty([0, 0])

        estimator = DecisionTreeClassifier(min_impurity_split=minimum_impurity_split)
        estimator.fit(df, classes)
        leafsIndexes = estimator.apply(df, check_input=True)

        for index, _ in df.iterrows():
            suma = 0
            value = 0
            for leafIndex, _ in df.iterrows():
                if leafsIndexes[index] == leafsIndexes[leafIndex]:
                    suma += 1
                    if classes[index] == classes[leafIndex]:
                        value += 1
            values = np.append(values, np.full((1, 1), (value / suma) * -1))
            # print("Count DCP for " + repr(index) + ". row of data.")

        return values
