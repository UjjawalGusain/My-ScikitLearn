import pandas as pd
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
from linear_regression import SGDRegressor as MySGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import time 

# Load the California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']

X=X.head(1000)
y = y[:1000]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=data.feature_names)

print(X.head())
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69, shuffle=True)

print("\n==== SGD REGRESSOR COMPARISON ====\n")

myLr = MySGDRegressor(step=0.001, epochs=1000, random_state=69, penalty='l2', alpha=0.05)

begin1 = time.time()
myLr.fit(X_train, y_train)
end1 = time.time()

sklearnLr = SklearnSGDRegressor(learning_rate='constant', eta0=0.001, max_iter=1000, alpha=0.05, shuffle=False, penalty='l2')

begin2 = time.time()
sklearnLr.fit(X_train, y_train)
end2 = time.time()
# Predictions
myPredictionsTest = myLr.predict(X_test)
sklearnPredictionsTest = sklearnLr.predict(X_test)

# Print results
# print(f"My coefficients: {myLr.coefficients}, bias: {myLr.bias}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, myPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, myPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, myPredictionsTest)}")
print(f"Time: ", end1-begin1)

print()

# Evaluate Sklearn SGDRegressor model
print(f"Sklearn coefficients: {sklearnLr.coef_}, bias: {sklearnLr.intercept_}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, sklearnPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, sklearnPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, sklearnPredictionsTest)}")
print(f"Time: ", end2-begin2)
