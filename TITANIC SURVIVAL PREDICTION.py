from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


# Load the dataset (change accordingly if you are in github)
data = pd.read_csv('E:\\CodSoft\\titanic\\Titanic.csv')



"""
    >>> print(data.head())
    >>> print(data.info())
    >>> print(data.describe())
"""



# Fill missing 'Age' values with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)


# Fill missing 'Embarked' values with the mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# Drop the 'Cabin' column due to a high number of missing values
data.drop(columns=['Cabin'], inplace=True)


# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop non-numeric columns
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)


"""
    >>> print(data.head())
"""


# Define feature variables and target variable
X = data.drop('Survived', axis=1)
y = data['Survived']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



# Predict on the test set
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
