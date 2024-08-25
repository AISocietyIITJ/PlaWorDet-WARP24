import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor

# Load the dataset
player_data = pd.read_csv('./Data/ProcessedData.csv')
player_data = player_data.dropna()
player_data.reset_index(drop=True, inplace=True)

# Prepare the features and target variable
X = player_data[['Predicted Price', 'Similarity Score']]
y = player_data['Actual Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble Method (Stacking)
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

# Prepare the output DataFrame for train and test predictions
train_predictions = stacking_model.predict(X_train)
test_predictions = y_pred_stack

# Create the DataFrame for train and test results
train_output_df = pd.DataFrame({
    'train_actual_price': y_train.values,
    'train_predicted_price': train_predictions
})

test_output_df = pd.DataFrame({
    'test_actual_price': pd.Series(y_test).reindex(X_test.index).values,
    'test_predicted_price': test_predictions
})

print(train_output_df.head())
print(test_output_df.head())

train_output_df.to_csv('./Data/train_output.csv')
test_output_df.to_csv('./Data/test_output.csv')
