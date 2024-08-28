import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the data
# Remove rows with missing values
data = data.replace('?', np.nan)
data = data.dropna()

# Drop unnecessary columns
data = data.drop(columns=['car name'])

# Convert 'horsepower' column to numeric
data['horsepower'] = pd.to_numeric(data['horsepower'])

# Split the data into features and target
X = data.drop(columns=['mpg'])
y = data['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the custom DecisionTree model
custom_tree = DecisionTree(criterion="information_gain", max_depth=5)
custom_tree.fit(X_train, y_train)
custom_predictions = custom_tree.predict(X_test)

# Train the scikit-learn DecisionTreeRegressor model
sklearn_tree = DecisionTreeRegressor(random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_predictions = sklearn_tree.predict(X_test)

# Evaluate the models
custom_rmse = rmse(y_test, custom_predictions)
sklearn_rmse = rmse(y_test, sklearn_predictions)

print("Custom DecisionTree RMSE:", custom_rmse)
print("Scikit-learn DecisionTreeRegressor RMSE:", sklearn_rmse)

