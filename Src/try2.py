import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import json


def complex_model(params, df, original_df):
    alpha, beta, gamma, delta, eta, zeta = params

    # Safeguard against invalid operations
    df = df.copy()
    df['Predicted Price'] = df['Predicted Price'].clip(lower=1e-8)  # Avoid zero or very small values
    df['Similarity Score'] = df['Similarity Score'].clip(lower=0)  # Ensure non-negative similarity scores

    # Calculate the final price using the complex formula
    try:
        df['Final Price'] = (alpha * (df['Predicted Price'] ** gamma) +
                             beta * (df['Similarity Score'] ** delta) *
                             (df['Predicted Price'] ** eta) + zeta)
    except Exception as e:
        print(f"Error in final price calculation: {e}")
        raise

    # Check for NaN values
    if df[['Predicted Price', 'Similarity Score', 'Final Price', 'Actual Price']].isnull().any().any():
        raise ValueError("DataFrame contains NaN values after computation.")

    # Ensure the lengths match
    if len(df) != len(original_df):
        raise ValueError(f"Length mismatch: df has {len(df)} rows, original_df has {len(original_df)} rows")

    # Calculate the mean absolute error between the final price and actual price
    error = mean_absolute_error(original_df['Actual Price'], df['Final Price'])
    return error


def optimize_parameters(df, original_df):
    # Initial guess for parameters [alpha, beta, gamma, delta, eta, zeta]
    initial_params = [1, 1, 1, 1, 1, 0]

    # Bounds for the parameters
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (None, None)]

    # Minimize the mean absolute error to find the best parameters
    result = minimize(complex_model, initial_params, args=(df, original_df), bounds=bounds, method='L-BFGS-B')

    # Extract the optimized parameters and the lowest error
    optimized_params = result.x
    lowest_error = result.fun

    return optimized_params, lowest_error


def save_results(df, filename, best_params):
    # Ensure 'Predicted Price' and 'Similarity Score' are not modified by the scaler
    df = df.copy()

    # Re-calculate Final Price using the best parameters
    alpha, beta, gamma, delta, eta, zeta = best_params
    df['Final Price'] = (alpha * (df['Predicted Price'] ** gamma) +
                         beta * (df['Similarity Score'] ** delta) *
                         (df['Predicted Price'] ** eta) + zeta)

    # Save DataFrame with Predicted Price, Actual Price, and Final Price
    df[['Predicted Price', 'Actual Price', 'Final Price']].to_csv(filename, index=False)


def save_parameters_to_json(params, filename):
    # Convert parameters to a dictionary
    params_dict = {
        'alpha': params[0],
        'beta': params[1],
        'gamma': params[2],
        'delta': params[3],
        'eta': params[4],
        'zeta': params[5]
    }

    # Save the dictionary to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(params_dict, json_file, indent=4)


# Example usage:
df = pd.read_csv('./Data/ProcessedData.csv')
original_df = df[['Predicted Price', 'Actual Price', 'Similarity Score']].copy()
df = df[['Predicted Price', 'Actual Price', 'Similarity Score']]
df = df.dropna()
df.reset_index(drop=True, inplace=True)
original_df = original_df.loc[df.index]  # Align original_df with df
print(df.describe())

# Find the best parameters
try:
    best_params, lowest_error = optimize_parameters(df, original_df)
    print(f"Best Parameters: {best_params}, Lowest MAE: {lowest_error}")

    # Save the result
    print('Saving Final Result')
    save_results(df, './result/Predicted_vs_Actual.csv', best_params)

    # Save best parameters to a JSON file
    print('Saving Best Parameters to JSON')
    save_parameters_to_json(best_params, './model/best_params.json')

except ValueError as e:
    print(f"Optimization failed: {e}")
