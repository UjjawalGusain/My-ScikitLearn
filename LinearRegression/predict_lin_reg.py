import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from linear_regression import LinearRegression as MyLinearRegression
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

print("\n==== LINEAR REGRESSOR COMPARISON ====\n")

myLinear = MyLinearRegression()
sklearnLinear = SklearnLinearRegression()

myLinear.fit(X_train, y_train)
sklearnLinear.fit(X_train, y_train)

myLinearPredTrain = myLinear.predict(X_train)
myLinearPredTest = myLinear.predict(X_test)

sklearnLinearPredTrain = sklearnLinear.predict(X_train)
sklearnLinearPredTest = sklearnLinear.predict(X_test)

# Evaluate My Linear Regressor
print(f"My Linear Regression Coefficients: {myLinear.coefficients}, Bias: {myLinear.bias}")
print(f"My Linear Regression - MSE: {mean_squared_error(y_test, myLinearPredTest)}")
print(f"My Linear Regression - MAE: {mean_absolute_error(y_test, myLinearPredTest)}")
print(f"My Linear Regression - R² Score: {r2_score(y_test, myLinearPredTest)}\n")

# Evaluate Sklearn Linear Regressor
print(f"Sklearn Linear Regression Coefficients: {sklearnLinear.coef_}, Bias: {sklearnLinear.intercept_}")
print(f"Sklearn Linear Regression - MSE: {mean_squared_error(y_test, sklearnLinearPredTest)}")
print(f"Sklearn Linear Regression - MAE: {mean_absolute_error(y_test, sklearnLinearPredTest)}")
print(f"Sklearn Linear Regression - R² Score: {r2_score(y_test, sklearnLinearPredTest)}\n")
