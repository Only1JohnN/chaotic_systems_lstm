# type: ignore tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(seq_length=10, input_dim=3):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, input_dim)),
        LSTM(50),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
