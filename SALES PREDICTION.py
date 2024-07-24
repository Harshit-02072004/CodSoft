import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
data = pd.read_csv('E:\\CodSoft\\sales\\advertising.csv')


"""
    # Display the first few rows
    >>> print(data.head())
"""


# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())


# Split the data into features and target
X = data.drop('Sales', axis=1)
y = data['Sales']


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')



# Plot both graphs in a single window using subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))


# Actual vs Predicted Sales plot
axes[0].scatter(y_test, y_pred, alpha=0.7, color='b')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title('Actual vs Predicted Sales')



# Feature Importance plot
coefficients = model.coef_
feature_importance = pd.Series(coefficients, index=X.columns).sort_values()

sns.barplot(x=feature_importance, y=feature_importance.index, ax=axes[1])
axes[1].set_xlabel('Coefficient')
axes[1].set_ylabel('Feature')
axes[1].set_title('Feature Importance')

plt.tight_layout()
plt.show()
