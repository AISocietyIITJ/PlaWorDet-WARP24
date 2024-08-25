import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import optuna

# Load the dataset
player_data = pd.read_csv('./Data/ProcessedData.csv')
player_data = player_data.dropna()
player_data.reset_index(drop=True, inplace=True)

# Prepare the features and target variable
X = player_data[['Predicted Price', 'Similarity Score']]
y = player_data['Actual Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f'Linear Regression MSE: {mse_linear}')

# 2. Polynomial Regression Model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f'Polynomial Regression MSE: {mse_poly}')

# 3. Support Vector Regression
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'Support Vector Regression MSE: {mse_svr}')

# 4. Gradient Boosting Regression
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost Regression MSE: {mse_xgb}')

# 5. Neural Network with TensorFlow/Keras
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_nn = nn_model.predict(X_test).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network MSE: {mse_nn}')

# 6. Regularization Techniques (Lasso and Ridge)
ridge_model = Ridge()
ridge_model.fit(X_train_poly, y_train)
y_pred_ridge = ridge_model.predict(X_test_poly)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'Ridge Regression MSE: {mse_ridge}')

lasso_model = Lasso(max_iter=10000)
lasso_model.fit(X_train_poly, y_train)
y_pred_lasso = lasso_model.predict(X_test_poly)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'Lasso Regression MSE: {mse_lasso}')

# 7. Ensemble Method (Stacking)
base_learners = [
    ('linear', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso(max_iter=10000)),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100))
]
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)
mse_stack = mean_squared_error(y_test, y_pred_stack)
print(f'Stacking Regressor MSE: {mse_stack}')

# 8. Hyperparameter Optimization with Optuna for Neural Network
def objective(trial):
    model = Sequential([
        Dense(trial.suggest_int('units1', 32, 128), activation='relu', input_shape=(X_train.shape[1],)),
        Dense(trial.suggest_int('units2', 16, 64), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    mse = model.evaluate(X_test, y_test, verbose=0)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
print(f'Best parameters: {study.best_params}')
print(f'Best MSE: {study.best_value}')
