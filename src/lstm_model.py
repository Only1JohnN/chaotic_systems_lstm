# type: ignore tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm(seq_length=10, input_dim=3):
    """
    Builds and compiles an LSTM model with dropout for regularization.
    :param seq_length: Length of the input sequence.
    :param input_dim: Number of features (3 for the Lorenz system).
    :return: Compiled Keras model.
    """
    model = Sequential([
        LSTM(32, input_shape=(seq_length, input_dim), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if __name__ == "__main__":
    model = build_lstm()
    model.summary()
