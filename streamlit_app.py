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

with st.expander("Fix Random Seed For Reproducibility"):
# fix random seed for reproducibility
    st.write('**Process**')
    tf.random.set_seed(7)


with st.expander("Get Stock Data"):
    # fix random seed for reproducibility
    tf.random.set_seed(7)

    # Define the stock ticker and load the data
    ticker = input('Enter stock ticker:')
    data = yf.download(ticker, start='2010-01-01', end='2024-01-01')

    # Use the 'Close' price for prediction
    df = data[['Close']]

    dataset = df.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)


    create_dataset(dataset)
