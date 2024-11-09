import streamlit as st
import base64

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

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
