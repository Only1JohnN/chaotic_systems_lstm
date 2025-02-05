# type: ignore tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

def build_rnn(seq_length=10, input_dim=3):
    """
    Builds and compiles a SimpleRNN model with dropout regularization.
    :param seq_length: Length of the input sequence.
    :param input_dim: Number of features (3 for the Lorenz system).
    :return: Compiled Keras model.
    """
    model = Sequential([
        SimpleRNN(32, input_shape=(seq_length, input_dim), return_sequences=True),
        Dropout(0.2),
        SimpleRNN(16),
        Dropout(0.2),
        Dense(3)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if __name__ == "__main__":
    model = build_rnn()
    model.summary()
