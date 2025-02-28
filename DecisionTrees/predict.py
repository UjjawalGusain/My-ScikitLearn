import pandas as pd
from decision_tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
train_df = pd.read_csv('train.csv')

train_label_mappings = {}

for col in train_df.columns:
    unique_values = train_df[col].unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    train_label_mappings[col] = mapping
    train_df[col] = train_df[col].map(mapping)

X = train_df.drop(columns=['Survived', 'Name', 'PassengerId', 'Ticket'])
y = train_df['Survived']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=69)

custom_dt = MyDecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=69)
custom_dt.fit(train_X, train_y)

# Sklearn Decision Tree
sklearn_dt = SklearnDecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=69)
sklearn_dt.fit(train_X, train_y)

custom_train_predictions = custom_dt.predict(train_X)
custom_test_predictions = custom_dt.predict(test_X)

sklearn_train_predictions = sklearn_dt.predict(train_X)
sklearn_test_predictions = sklearn_dt.predict(test_X)

# Compute metrics for custom Decision Tree
custom_train_accuracy = accuracy_score(train_y, custom_train_predictions)
custom_test_accuracy = accuracy_score(test_y, custom_test_predictions)

custom_train_precision = precision_score(train_y, custom_train_predictions)
custom_test_precision = precision_score(test_y, custom_test_predictions)

custom_train_recall = recall_score(train_y, custom_train_predictions)
custom_test_recall = recall_score(test_y, custom_test_predictions)

custom_train_f1 = f1_score(train_y, custom_train_predictions)
custom_test_f1 = f1_score(test_y, custom_test_predictions)

print(f"Custom Train Accuracy: {custom_train_accuracy:.4f}")
print(f"Custom Test Accuracy: {custom_test_accuracy:.4f}")
print(f"Custom Train Precision: {custom_train_precision:.4f}")
print(f"Custom Test Precision: {custom_test_precision:.4f}")
print(f"Custom Train Recall: {custom_train_recall:.4f}")
print(f"Custom Test Recall: {custom_test_recall:.4f}")
print(f"Custom Train F1 Score: {custom_train_f1:.4f}")
print(f"Custom Test F1 Score: {custom_test_f1:.4f}")

print("------------------------------------------")
print("------------------------------------------")
print("------------------------------------------")

# Compute metrics for sklearn Decision Tree
sklearn_train_accuracy = accuracy_score(train_y, sklearn_train_predictions)
sklearn_test_accuracy = accuracy_score(test_y, sklearn_test_predictions)

sklearn_train_precision = precision_score(train_y, sklearn_train_predictions)
sklearn_test_precision = precision_score(test_y, sklearn_test_predictions)

sklearn_train_recall = recall_score(train_y, sklearn_train_predictions)
sklearn_test_recall = recall_score(test_y, sklearn_test_predictions)

sklearn_train_f1 = f1_score(train_y, sklearn_train_predictions)
sklearn_test_f1 = f1_score(test_y, sklearn_test_predictions)

print(f"Sklearn Train Accuracy: {sklearn_train_accuracy:.4f}")
print(f"Sklearn Test Accuracy: {sklearn_test_accuracy:.4f}")
print(f"Sklearn Train Precision: {sklearn_train_precision:.4f}")
print(f"Sklearn Test Precision: {sklearn_test_precision:.4f}")
print(f"Sklearn Train Recall: {sklearn_train_recall:.4f}")
print(f"Sklearn Test Recall: {sklearn_test_recall:.4f}")
print(f"Sklearn Train F1 Score: {sklearn_train_f1:.4f}")
print(f"Sklearn Test F1 Score: {sklearn_test_f1:.4f}")