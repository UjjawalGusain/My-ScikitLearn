import numpy as np

class LinearRegression:
    def __init__(self):
        self.bias = None
        self.coefficients = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = np.column_stack((np.ones(X.shape[0]), X))

        X_transpose = X.T
        inverse = np.linalg.pinv(X_transpose @ X)
        b = inverse @ X_transpose @ y

        self.bias = b[0]
        self.coefficients = b[1:]

    def predict(self, X):
        assert self.coefficients is not None and self.bias is not None, "Fit the model first"

        X = np.array(X)
        return np.dot(X, self.coefficients) + self.bias
