import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from linear_regression import LinearRegression as MyLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')

df = df.dropna()
df = df.drop(columns=['Job Title'])

label_mappings = {}

for col in df.columns:
    if col in ['Age', 'Salary', 'Years of Experience']:
        continue
    unique_values = df[col].unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    label_mappings[col] = mapping
    df[col] = df[col].map(mapping)

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69, shuffle=True)
# print(X_train.head())


myLr = MyLinearRegression()
sklearnLr = SklearnLinearRegression()




myLr.fit(X_train, y_train)
sklearnLr.fit(X_train, y_train)


myPredictionsTrain = myLr.predict(X_train)
myPredictionsTest = myLr.predict(X_test)

sklearnPredictions = sklearnLr.predict(X_train)
sklearnPredictionsTest = sklearnLr.predict(X_test)

print(f"My coefficients: {myLr.coefficients}, bias: {myLr.bias}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, myPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, myPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, myPredictionsTest)}")

# Evaluate Sklearn LinearRegression model
print(f"Sklearn coefficients: {sklearnLr.coef_}, bias: {sklearnLr.intercept_}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, sklearnPredictionsTest)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, sklearnPredictionsTest)}")
print(f"R² Score: {r2_score(y_test, sklearnPredictionsTest)}")


