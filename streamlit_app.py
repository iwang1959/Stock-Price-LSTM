import streamlit as st
import base64

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://media.istockphoto.com/id/1310618429/photo/price-of-btc-is-going-to-breakout.jpg?s=612x612&w=0&k=20&c=46lAhJApOlBF32MEZskG8E1HO18JDDLyRPfg9NX_KoE=');
        background-size: cover;
        background-position: center;
    }
    
    h1, h2, h3, h4, h5, h6, p, div {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader('📈 Recurrent Neural Network (RNN)')

st.info('This application uses the Long short-term memory (LSTM) to determine stock price.')

st.write('RECURRENT NEURAL NETWORK')

st.markdown("""
<u>Key Features of RNN:</u>

- One of the oldest neural network architectures, developed in the 1980s.
- Recently became popular due to the availability of large datasets and increased computational power.
- Primarily used for **sequence** and **time series data**.
- RNNs have internal memory, allowing them to store information from previous steps in the sequence.
- The internal memory helps the network make predictions based on historical data.
- RNNs are **feedback neural networks**, with self-loops at the hidden layer.
""", unsafe_allow_html=True)
