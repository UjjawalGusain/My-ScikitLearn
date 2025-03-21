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
    
class Ridge:
    def __init__(self, alpha=1.0):
        self.bias = None
        self.coefficients = None
        self.alpha = alpha

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = np.column_stack((np.ones(X.shape[0]), X))
        I = np.eye(X.shape[1])
        I[0][0] = 0

        X_transpose = X.T
        XT_X = X_transpose @ X
        alpha_I = self.alpha * I
        inverse = np.linalg.pinv(XT_X + alpha_I)
        b = inverse @ X_transpose @ y

        self.bias = b[0]
        self.coefficients = b[1:]

    def predict(self, X):
        assert self.coefficients is not None and self.bias is not None, "Fit the model first"

        X = np.array(X)
        return np.dot(X, self.coefficients) + self.bias


class SGDRegressor:
    def __init__(self, step=1e-3, epochs=1000, batch_size=1, random_state=None, penalty=None, alpha=1):
        self.bias = None
        self.coefficients = None
        self.step = step
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()

        X = np.column_stack((np.ones(X.shape[0]), X))
        num_features = X.shape[1]

        self.coefficients = np.ones(num_features)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])  

            for i in range(0, X.shape[0], self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                y_hat = X_batch @ self.coefficients  
                gradient = (X_batch.T @ (y_hat - y_batch)) / self.batch_size  
                
                if self.penalty == 'l2':
                    gradient += self.alpha * self.coefficients 

                self.coefficients -= self.step * gradient 

        self.bias = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        assert self.coefficients is not None and self.bias is not None, "Fit the model first"

        X = np.array(X)
        return np.dot(X, self.coefficients) + self.bias
