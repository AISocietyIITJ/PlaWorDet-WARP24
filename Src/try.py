import pandas as pd
import numpy as np
from scipy.optimize import minimize


def mse_loss(params, predicted_price, similarity_score, actual_price):
    alpha, beta = params
    predicted_final_price = alpha * predicted_price + beta * similarity_score * predicted_price
    mse = np.mean((actual_price - predicted_final_price) ** 2)
    return mse


def find_best_alpha_beta(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    # Extract necessary columns
    predicted_price = df['Predicted Price'].values
    similarity_score = df['Similarity Score'].values
    actual_price = df['Actual Price'].values

    # Initial guess for alpha and beta
    initial_guess = [1.0, 1.0]

    # Minimize the MSE loss function
    result = minimize(
        mse_loss,
        initial_guess,
        args=(predicted_price, similarity_score, actual_price),
        method='BFGS'
    )

    # Extract optimal alpha and beta
    alpha, beta = result.x

    return alpha, beta


def calculate_final_loss(file_path, alpha, beta):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    predicted_price = df['Predicted Price'].values
    similarity_score = df['Similarity Score'].values
    actual_price = df['Actual Price'].values

    # Calculate the predicted final price using the optimal alpha and beta
    predicted_final_price = alpha * predicted_price + beta * similarity_score * predicted_price

    # Calculate the Mean Squared Error
    mse = np.mean((actual_price - predicted_final_price) ** 2)

    return mse


# Example usage:
file_path = './Data/ProcessedData.csv'
alpha, beta = find_best_alpha_beta(file_path)
print(f"Optimal Alpha: {alpha:.4f}, Optimal Beta: {beta:.4f}")
print(f"Final Loss: {calculate_final_loss(file_path, alpha, beta):.4f}")
