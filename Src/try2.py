import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

    # Debugging information
    # print(df[['Predicted Price', 'Similarity Score', 'Final Price', 'Actual Price']].describe())

    # Check for NaN values
    if df[['Predicted Price', 'Similarity Score', 'Final Price', 'Actual Price']].isnull().any().any():
        raise ValueError("DataFrame contains NaN values after computation.")

    # Calculate the mean absolute error between the final price and actual price
    error = mean_absolute_error(original_df['Actual Price'], df['Final Price'])
    return error

def optimize_parameters(df, original_df, scaler):
    # Initial guess for parameters [alpha, beta, gamma, delta, eta, zeta]
    initial_params = [1, 1, 1, 1, 1, 0]

    # Bounds for the parameters
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (None, None)]

    # Minimize the mean squared error to find the best parameters
    result = minimize(complex_model, initial_params, args=(df, original_df), bounds=bounds, method='L-BFGS-B')

    # Extract the optimized parameters and the lowest error
    optimized_params = result.x
    lowest_error = result.fun

    return optimized_params, lowest_error

def save_results(df, filename, scaler, original_df):
    # Inverse transform the columns
    df[['Predicted Price', 'Actual Price', 'Final Price']] = scaler.inverse_transform(df[['Predicted Price', 'Actual Price', 'Similarity Score']])

    df.rename(columns={'Actual Price': 'Final Price', 'Final Price': 'Actual Price'}, inplace=True)


    df['Actual Price'] = scaler.inverse_transform(df['Actual Price'])

    df.rename(columns={'Actual Price': 'Final Price', 'Final Price': 'Actual Price'}, inplace=True)

    # Re-calculate Final Price using the original values
    df['Final Price'] = (df['Predicted Price'] ** best_params[2] * best_params[0] +
                         df['Similarity Score'] ** best_params[3] * df['Predicted Price'] ** best_params[4] * best_params[1] +
                         best_params[5])

    # Save DataFrame with Predicted Price, Actual Price, and Final Price
    df[['Predicted Price', 'Actual Price', 'Final Price']].to_csv(filename, index=False)

# Example usage:
df = pd.read_csv('./Data/ProcessedData.csv')
original_df = df[['Predicted Price', 'Actual Price', 'Similarity Score']].copy()
df = df[['Predicted Price', 'Actual Price', 'Similarity Score']]
df = df.dropna()
df.reset_index(drop=True, inplace=True)
print(df.describe())
print('----------------------------------------------------')

scalar = StandardScaler()
df_scaled = scalar.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=['Predicted Price', 'Actual Price', 'Similarity Score'])
df.reset_index(drop=True, inplace=True)
print(df.describe())

# Find the best parameters
try:
    best_params, lowest_error = optimize_parameters(df, original_df, scalar)
    print(f"Best Parameters: {best_params}, Lowest MAE: {lowest_error}")

    # Re-calculate Final Price with the best parameters
    alpha, beta, gamma, delta, eta, zeta = best_params
    df['Predicted Price'] = df['Predicted Price'].clip(lower=1e-8)
    df['Similarity Score'] = df['Similarity Score'].clip(lower=0)
    df['Final Price'] = (alpha * (df['Predicted Price'] ** gamma) +
                         beta * (df['Similarity Score'] ** delta) *
                         (df['Predicted Price'] ** eta) + zeta)

    # Save the result
    print('Saving Final Result')
    save_results(df, './result/Predicted_vs_Actual.csv', scalar, original_df)

except ValueError as e:
    print(f"Optimization failed: {e}")
