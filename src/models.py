import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import score as prep
df, X, y = prep.main()


# 1) LINEAR REGRESSION

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating and training the linear regression model
lr_model = LinearRegression()

# Perform cross-validation on the training data
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)


print("\nCross-validation RMSE scores:", cv_rmse_scores)
print("Mean CV RMSE: {:.4f}".format(cv_rmse_scores.mean()))
print("Standard Deviation of CV RMSE: {:.4f}".format(cv_rmse_scores.std()))

# training final model on training set
lr_model.fit(X_train, y_train)

# evaluating on test set
y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Results ---:")
print("Mean Squared Error: {:.4f}".format(mse))
print("Root Mean Squared Error: {:.4f}".format(rmse))
print("Mean Absolute Error: {:.4f}".format(mae))
print("R-squared Score: {:.4f}".format(r2))


# 2) RANDOM FOREST REGRESSOR

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print("\nCross-validation RMSE scores:", cv_rmse_scores)
print("Mean CV RMSE: {:.4f}".format(cv_rmse_scores.mean()))
print("Standard Deviation of CV RMSE: {:.4f}".format(cv_rmse_scores.std()))

# Train final model on entire training set
rf_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Evaluation Results ---:")
print("\nFinal Evaluation on Test Set:")
print("Mean Squared Error: {:.4f}".format(mse))
print("Root Mean Squared Error: {:.4f}".format(rmse))
print("Mean Absolute Error: {:.4f}".format(mae))
print("R-squared Score: {:.4f}".format(r2))




# 3) GRADIENT BOOSTING MACHINE

















