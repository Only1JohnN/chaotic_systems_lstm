# type: ignore tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def build_rnn(seq_length=10, input_dim=3):
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=(seq_length, input_dim)),
        SimpleRNN(50),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
