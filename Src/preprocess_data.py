import numpy as np
import pandas as pd
from price_predictor import predictor
from Similarity_Score import sim_func

def make_data(player_data):

    base_price_predictor = predictor()

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['ID', 'Name', 'Club', 'Similarity Score', 'Predicted Price', 'Actual Price'])

    for index, row in player_data.iterrows():
        # Calculate base price and similarity score
        predicted_base_price, actual_price = base_price_predictor(index)

        try:
            similarity_score = sim_func(int(row['ID']), str(row['Club']))

            # Append the result to the DataFrame
            results_df = results_df._append({
                'ID': row['ID'],
                'Name': row['Name'],
                'Club': row['Club'],
                'Similarity Score': similarity_score,
                'Predicted Price': predicted_base_price,
                'Actual Price': actual_price
            }, ignore_index=True)
        except ValueError:
            continue

    return results_df

# Load player data
player_data = pd.read_csv('./Data/MergedData.csv')
player_data.reset_index(drop=True, inplace=True)

# Generate and store the data in a DataFrame
results_df = make_data(player_data)

# Optionally, save the DataFrame to a CSV file
results_df.to_csv('./Data/ProcessedData.csv', index=False)
