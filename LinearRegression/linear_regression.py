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


class SGDRegressor:
    def __init__(self, step=1e-3, epochs=1000):
        self.bias = None
        self.coefficients = None
        self.step = step
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        self.bias = 0
        self.coefficients = np.ones(X.shape[1])

        for epoch in range(self.epochs): 

            indices = np.arange(X.shape[0])  
            np.random.shuffle(indices)
            for i in indices:
                y_hat = np.dot(X[i], self.coefficients) + self.bias
                slope_bias = y_hat - y[i]
                self.bias = self.bias - self.step * slope_bias

                slope_coefficients = (y_hat - y[i])* X[i]
                self.coefficients = self.coefficients - self.step * slope_coefficients


    def predict(self, X):
        assert self.coefficients is not None and self.bias is not None, "Fit the model first"

        X = np.array(X)
        return np.dot(X, self.coefficients) + self.bias

