import streamlit as st
from price_predictor import predictor

# Streamlit app
def main():
    pred = predictor()

    st.title("Player Information Processing")

    # Input fields
    player_name = st.text_input("Enter Player Name")
    club_name = st.text_input("Enter Club Name")
    player_position = st.text_input("Enter Player Position")

    # Button to trigger the function
    if st.button("Submit"):
        result = pred(player_name, club_name, player_position)
        st.write(result)

if __name__ == "__main__":
    main()
