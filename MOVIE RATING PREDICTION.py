import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
data = pd.read_csv('E:\\CodSoft\\iris\\IRIS.csv')
# print(data)


"""
    >>> print(data.head())
"""


# Split the data into features and target
X = data.drop('species', axis=1)
y = data['species']


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train a Support Vector Machine classifier
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
