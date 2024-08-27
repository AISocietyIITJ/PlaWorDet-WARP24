import streamlit as st
from price_predictor import predictor

# Set custom CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #000000; /* Black background */
        color: #ffffff; /* White text */
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background-color: #333333; /* Dark gray input fields */
        color: #ffffff; /* White text in input fields */
        border-radius: 10px;
        border: 1px solid #555555; /* Slightly lighter border */
        padding: 10px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #555555; /* Gray button */
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #777777; /* Lighter gray on hover */
    }
    .stMarkdown {
        text-align: center;
        color: #ffffff; /* White text */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
def main():

    pred = predictor()
    st.markdown("# Player Price Prediction", unsafe_allow_html=True)

    st.markdown("### Enter Player Information:", unsafe_allow_html=True)

    # Input fields
    player_name = st.text_input("Enter Player Name")
    club_name = st.text_input("Enter Club Name")
    player_position = st.text_input("Enter Player Position")


    # Button to trigger the function
    if st.button("Submit"):
        result = pred(player_name, club_name, player_position)  # Replace with: pred(player_name, club_name, player_position)
        st.markdown(f"### Predicted Price: {result:.2f} Million Euros", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
