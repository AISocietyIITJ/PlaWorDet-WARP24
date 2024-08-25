import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def adjust_player_price(base_price, similarity_score, alpha):
    """
    Adjust the player's base price based on similarity score.
    """
    return base_price * (1 + alpha * (similarity_score - 2))


def grid_search_alpha(player_data, alpha_values):

    best_alpha = None
    best_error = float('inf')

    for alpha in alpha_values:
        adjusted_prices = []
        actual_prices = []

        for index, row in player_data.iterrows():
            # Calculate base price and similarity score
            predicted_base_price, actual_price = row['Predicted Price'], row['Actual Price']
            similarity_score = row['Similarity Score']

            # Adjust the price using the current alpha
            adjusted_price = adjust_player_price(predicted_base_price, similarity_score, alpha)

            # Append to lists
            adjusted_prices.append(adjusted_price)
            actual_prices.append(actual_price)

        # Calculate error for the current alpha
        error = mean_absolute_error(actual_prices, adjusted_prices)

        # Update best alpha if this one is better
        if error < best_error:
            best_error = error
            best_alpha = alpha

        print(f'Alpha: {alpha}, MAE: {error}')

    return best_alpha, best_error


# Example usage
alpha_values = np.linspace(0, 1, 50)  # Grid of 50 alpha values between 0 and 1
player_data = pd.read_csv('./Data/ProcessedData.csv')
player_data = player_data.dropna()
player_data.reset_index(drop=True, inplace=True)

print('intial error - ', mean_absolute_error(player_data['Predicted Price'], player_data['Actual Price']))

best_alpha, best_error = grid_search_alpha(player_data, alpha_values)
print(f'Best Alpha: {best_alpha}, Best MAE: {best_error}')
