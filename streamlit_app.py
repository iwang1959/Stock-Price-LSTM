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
from tensorflow.keras.models import load_model
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


# Load the saved LSTM model
@st.cache_resource
def load_lstm_model():
    model = load_model('lstm_model.weights.h5')
    return model

model = load_lstm_model()

# Function to make predictions
def make_predictions(input_data, look_back=1):
    # Load the scaler used during training
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    # Prepare the data for the LSTM model
    dataX = []
    for i in range(len(input_data) - look_back - 1):
        dataX.append(input_data[i:(i + look_back), 0])
    dataX = np.array(dataX)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    # Make predictions
    predictions = model.predict(dataX)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Streamlit UI
st.title('Stock Price Prediction using LSTM')
st.write('This app uses a pre-trained LSTM model to predict stock prices.')

# File upload for input data
uploaded_file = st.file_uploader("Upload your stock data (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data.head())

    # Ensure the data is in the correct format
    if 'Close' in data.columns:
        # Prepare the input data (using 'Close' prices)
        input_data = data[['Close']].values

        # Make predictions
        predictions = make_predictions(input_data)

        # Display predictions
        st.write("Predicted Prices:")
        st.line_chart(predictions)
    else:
        st.error("The uploaded CSV file must contain a 'Close' column for stock prices.")
