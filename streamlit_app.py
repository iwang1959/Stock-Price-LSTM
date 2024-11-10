import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Streamlit App
st.title('Stock Price Prediction using LSTM')
st.write('This app predicts the stock prices using an LSTM model.')

# User input for the stock ticker
ticker = st.text_input('Enter stock ticker (e.g., AAPL, GOOG, TSLA):', 'AAPL')

# Load stock data
if ticker:
    try:
        data = yf.download(ticker, start='2010-01-01', end='2024-01-01')

        # Check if data is retrieved
        if not data.empty:
            st.write(f"Data for {ticker}:")
            st.dataframe(data.tail())

            # Use the 'Close' price for prediction
            df = data[['Close']]

            # Plot the closing price
            st.line_chart(df)

            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df)

            # Prepare the input data for LSTM
            X, y = [], []
            window_size = 10

            for i in range(len(scaled_data) - window_size):
                X.append(scaled_data[i:i + window_size])
                y.append(scaled_data[i + window_size])

            # Convert lists to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Reshape the input to be [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mse')

            # Train the model
            st.write("Training the model. Please wait...")
            model.fit(X, y, epochs=50, verbose=0)

            # Make predictions
            predicted = model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted)

            # Plot the predictions
            st.subheader('Predicted Prices (last 50 predictions):')
            prediction_df = pd.DataFrame(predicted_prices[-50:], columns=['Predicted Price'])
            st.line_chart(prediction_df)

            # Display the first 5 predictions
            st.write("First 5 Predictions:", predicted_prices[:5].flatten())

        else:
            st.error("No data found for the given ticker. Please try a different one.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
