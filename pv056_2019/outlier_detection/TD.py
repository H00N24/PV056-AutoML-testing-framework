import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TDMetric:
    def findLeafDepthWithoutPrunning(self, df, classes):

        values = np.empty([0, 0])

        estimator = DecisionTreeClassifier()
        estimator.fit(df, classes)

        n_nodes = estimator.tree_.node_count
        # print(n_nodes)
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)

        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))

        leafsIndexes = estimator.apply(df, check_input=True)
        for index in leafsIndexes:
            values = np.append(values, np.full((1, 1), node_depth[index]))

        return values

    def findLeafDepthWithPrunning(self, df, classes, minimum_impurity_split):

        values = np.empty([0, 0])

        estimator = DecisionTreeClassifier(min_impurity_split=minimum_impurity_split)
        estimator.fit(df, classes)

        n_nodes = estimator.tree_.node_count
        # print(n_nodes)
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)

        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))

        leafsIndexes = estimator.apply(df, check_input=True)
        for index in leafsIndexes:
            values = np.append(values, np.full((1, 1), node_depth[index]))

        return values
