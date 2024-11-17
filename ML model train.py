# /mnt/data/tune_rf_regressor.py
# This model training python file can be used to train parameters. You can alter the numbers to achieve the desired outcome.
# Dinalofcl - 2024

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = '/Users/sanjulaweerasekara/Desktop/Programming Project/EnergyDataSET1.csv'
data = pd.read_csv(file_path)

# Assuming 'target' is the name of the column you're predicting
# Adjust this based on your dataset
X = data.drop(columns=['timestamps'])  # Features
y = data['consumption']                 # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestRegressor(random_state=42)

# Change the values according to you
param_grid = {
    'n_estimators': [50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')


# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate on the test set
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("Best Parameters:", best_params)
print("Training R^2 Score:", train_r2)
print("Test R^2 Score:", test_r2)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
