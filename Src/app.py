import streamlit as st
import torch
from price_predictor import predictor  # Update this import based on your actual module name
from PIL import Image

def main():
    st.title("Player Auction Price Predictor")

    # Load the predictor model
    pred = predictor()

    # User input for player name
    player_name = st.text_input("Enter the player's name:")

    if st.button("Predict Price"):
        if player_name:
            try:
                actual_worth, predicted_worth = pred(player_name)
                st.write(f"Predicted Auction Price for {player_name}: {predicted_worth:.2f}")
                st.write(f"Actual Auction Price for {player_name}: {actual_worth}")
            except ValueError as e:
                st.error(e)
        else:
            st.error("Please enter a player's name.")

    image = Image.open('./result/result.png')
    st.image(image)

if __name__ == "__main__":
    main()
