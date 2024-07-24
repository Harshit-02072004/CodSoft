import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("E:\\CodSoft\\titanic\\Movies.csv")


"""
    Display the first few rows
    >>> print(data.head())

    Check for missing values and data types
    >>> print(data.info())

    Summary statistics for numeric columns
    >>> print(data.describe())

"""


# Drop rows where 'Rating' is missing (since that's our target variable)
data = data.dropna(subset=['Rating'])


# Convert 'Year' to a float
data['Year'] = data['Year'].str.extract('(\d+)').astype(float)


# Convert 'Duration' to a float (remove ' min' and convert to numeric)
data['Duration'] = data['Duration'].str.replace(' min', '').astype(float)


# Convert 'Votes' to numeric (remove commas and convert to float)
data['Votes'] = data['Votes'].str.replace(',', '').astype(float)


# Fill missing numeric values with the median of the column
numeric_cols = ['Year', 'Duration', 'Votes']
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())


# Fill missing string values with a placeholder
string_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
data[string_cols] = data[string_cols].fillna('Unknown')


# Display the cleaned data
print(data.head())


# One-hot encode categorical features
genre_dummies = data['Genre'].str.get_dummies(sep=', ')
director_dummies = pd.get_dummies(data['Director'], prefix='Director')
actor1_dummies = pd.get_dummies(data['Actor 1'], prefix='Actor1')
actor2_dummies = pd.get_dummies(data['Actor 2'], prefix='Actor2')
actor3_dummies = pd.get_dummies(data['Actor 3'], prefix='Actor3')


# Combine all features into a single DataFrame
features = pd.concat(
    [
        data[['Year', 'Duration', 'Votes']],
        genre_dummies,
        director_dummies,
        actor1_dummies,
        actor2_dummies,
        actor3_dummies,
    ],
    axis=1,
)


# Define the target variable
target = data['Rating']


# Display the feature set
print(features.head())


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Predict on the test set
y_pred = model.predict(X_test)


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')


# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2:.2f}')
