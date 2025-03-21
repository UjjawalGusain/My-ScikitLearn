import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge as SklearnRidge
from linear_regression import Ridge as MyRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']
df = df.dropna()


X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69, shuffle=True)

print("\n==== RIDGE REGRESSOR COMPARISON ====\n")

myRidgeRegression = MyRidge(alpha=0.5)
sklearnRidgeRegression = SklearnRidge(alpha=0.5)

myRidgeRegression.fit(X_train, y_train)
sklearnRidgeRegression.fit(X_train, y_train)

myRidgePredTrain = myRidgeRegression.predict(X_train)
myRidgePredTest = myRidgeRegression.predict(X_test)

sklearnRidgePredTrain = sklearnRidgeRegression.predict(X_train)
sklearnRidgePredTest = sklearnRidgeRegression.predict(X_test)

# Evaluate My Ridge Regressor
print(f"My Ridge Regression Coefficients: {myRidgeRegression.coefficients}, Bias: {myRidgeRegression.bias}")
print(f"My Ridge Regression - MSE: {mean_squared_error(y_test, myRidgePredTest)}")
print(f"My Ridge Regression - MAE: {mean_absolute_error(y_test, myRidgePredTest)}")
print(f"My Ridge Regression - R² Score: {r2_score(y_test, myRidgePredTest)}\n")

# Evaluate Sklearn Ridge Regressor
print(f"Sklearn Ridge Regression Coefficients: {sklearnRidgeRegression.coef_}, Bias: {sklearnRidgeRegression.intercept_}")
print(f"Sklearn Ridge Regression - MSE: {mean_squared_error(y_test, sklearnRidgePredTest)}")
print(f"Sklearn Ridge Regression - MAE: {mean_absolute_error(y_test, sklearnRidgePredTest)}")
print(f"Sklearn Ridge Regression - R² Score: {r2_score(y_test, sklearnRidgePredTest)}\n")
