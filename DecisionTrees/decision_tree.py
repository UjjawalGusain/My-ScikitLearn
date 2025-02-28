import numpy as np

class DecisionTreeClassifier:

    class TreeNode:
        def __init__(self, best_feature=None, left=None, right=None, best_threshold=0, label=None):
            self.best_feature = best_feature
            self.left = left
            self.right = right
            self.best_threshold = best_threshold
            self.label = label  

    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None):
        self.criterion = criterion  
        self.splitter = splitter  
        self.max_depth = max_depth  
        self.min_samples_split = min_samples_split  
        self.min_samples_leaf = min_samples_leaf  
        self.max_features = max_features  
        self.random_state = random_state  
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease  
        self.class_weight = class_weight  
        self.root = None
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _find_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        return -np.sum(probabilities * np.log2(probabilities))

    def _find_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / np.sum(counts)
        return 1 - np.sum(probabilities ** 2)

    def _calculate_impurity(self, y):
        if self.criterion == 'entropy':
            return self._find_entropy(y)
        else:
            return self._find_gini(y)

    def _split(self, column, threshold):
        left_indices = np.argwhere(column <= threshold).flatten()
        right_indices = np.argwhere(column > threshold).flatten()
        return left_indices, right_indices

    def _information_gain(self, column, y, threshold):
        parent_impurity = self._calculate_impurity(y)
        left_indices, right_indices = self._split(column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        left_impurity = self._calculate_impurity(y[left_indices])
        right_impurity = self._calculate_impurity(y[right_indices])

        total = len(column)
        children_impurity = (len(left_indices) / total) * left_impurity + (len(right_indices) / total) * right_impurity

        return parent_impurity - children_impurity

    def _best_split(self, X, y, feature_indices):
        best_threshold, best_index, best_gain = None, None, -np.inf  

        for index in feature_indices:
            column = X[:, index]
            thresholds = np.unique(column)

            for threshold in thresholds:
                gain = self._information_gain(column, y, threshold)

                # applying min_impurity_decrease
                if gain > best_gain and gain > self.min_impurity_decrease: 
                    best_gain = gain
                    best_index = index
                    best_threshold = threshold

        return best_index, best_threshold

    def _build_tree(self, X, y, depth=0):
        labels, counts = np.unique(y, return_counts=True)

        if len(labels) == 1:
            return self.TreeNode(label=labels[0])

        # applying max_depth
        if self.max_depth is not None and depth >= self.max_depth:  
            majority_class = labels[np.argmax(counts)]
            return self.TreeNode(label=majority_class)

        feature_indices = np.random.choice(X.shape[1], self.max_features or X.shape[1], replace=False)
        best_feature_index, best_threshold = self._best_split(X, y, feature_indices)

        if best_feature_index is None:
            majority_class = labels[np.argmax(counts)]
            return self.TreeNode(label=majority_class)

        left_indices, right_indices = self._split(X[:, best_feature_index], best_threshold)

        # applying min_samples_leaf
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:  
            majority_class = labels[np.argmax(counts)]
            return self.TreeNode(label=majority_class)

        # applying min_samples_split
        if len(left_indices) + len(right_indices) < self.min_samples_split:  
            majority_class = labels[np.argmax(counts)]
            return self.TreeNode(label=majority_class)

        left_tree = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return self.TreeNode(best_feature_index, left_tree, right_tree, best_threshold)

    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.max_features is None:
            self.max_features = X.shape[1]
        else:
            self.max_features = min(self.max_features, X.shape[1])

        self.root = self._build_tree(X, y)

    def _predict_single(self, x, node):
        if node.label is not None:
            return node.label

        if x[node.best_feature] <= node.best_threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        X = X.to_numpy()
        return np.array([self._predict_single(x, self.root) for x in X])
