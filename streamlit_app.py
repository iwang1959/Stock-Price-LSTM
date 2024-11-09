import streamlit as st
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://wallpapers.com/images/hd/stock-market-drastic-market-movement-baglidta60etpuu7.jpg');
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
