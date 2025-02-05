import numpy as np
from src.data_preprocessing import preprocess_data
from src.lstm_model import build_lstm
from src.rnn_model import build_rnn
import os

# Load processed data
data = np.load("data/processed_data.npz")
X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

# Train LSTM
lstm = build_lstm()
lstm.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
os.makedirs("saved_models", exist_ok=True)
lstm.save("saved_models/lstm_model.h5")

# Train RNN
rnn = build_rnn()
rnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
rnn.save("saved_models/rnn_model.h5")
