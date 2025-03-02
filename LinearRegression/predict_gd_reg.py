import pandas as pd
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from linear_regression import SGDRegressor as MySGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69, shuffle=True)

print("\n==== SGD REGRESSOR COMPARISON ====\n")

myLr = MySGDRegressor(step=0.01, epochs=100)
myLr.fit(X_train, y_train)

sklearnLr = SklearnSGDRegressor(learning_rate='constant', max_iter=1000, alpha=0, shuffle=False)
sklearnLr.fit(X_train, y_train)

# Predictions
myPredictionsTest = myLr.predict(X_test)
sklearnPredictionsTest = sklearnLr.predict(X_test)

# Print results
print(f"My coefficients: {myLr.coefficients}, bias: {myLr.bias}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, myPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, myPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, myPredictionsTest)}")

print()

# Evaluate Sklearn SGDRegressor model
print(f"Sklearn coefficients: {sklearnLr.coef_}, bias: {sklearnLr.intercept_}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, sklearnPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, sklearnPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, sklearnPredictionsTest)}")
