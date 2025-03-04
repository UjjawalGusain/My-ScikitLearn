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
    def __init__(self, step=1e-3, epochs=1000, batch_size=1, random_state = None):
        self.bias = None
        self.coefficients = None
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        self.bias = 0
        self.coefficients = np.ones(X.shape[1])
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for epoch in range(self.epochs): 
            for batch in range(X.shape[0] // self.batch_size):
                batch_indices = np.random.choice(X.shape[0], self.batch_size, replace=False)

                y_hat = np.dot(X[batch_indices], self.coefficients) + self.bias
                slope_bias = y_hat - y[batch_indices]
                self.bias -= self.step * np.mean(slope_bias)

                slope_coefficients = slope_bias[:, np.newaxis] * X[batch_indices]
                self.coefficients -= self.step * np.mean(slope_coefficients, axis=0)

    def predict(self, X):
        assert self.coefficients is not None and self.bias is not None, "Fit the model first"

        X = np.array(X)
        return np.dot(X, self.coefficients) + self.bias