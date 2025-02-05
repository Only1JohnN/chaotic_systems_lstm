# type: ignore tensorflow

import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
from src.data_preprocessing import preprocess_data
from src.lstm_model import build_lstm
from src.rnn_model import build_rnn
import pandas as pd

# Ensure Lorenz data exists; if not, generate it.
if not os.path.exists("data/lorenz_data.csv"):
    from src.lorenz import generate_lorenz_data
    print("📌 Lorenz data not found. Generating data...")
    t, data = generate_lorenz_data()
    import pandas as pd
    df = pd.DataFrame(data, columns=["x", "y", "z"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/lorenz_data.csv", index=False)
    print("✅ Lorenz data saved to 'data/lorenz_data.csv'")

# Preprocess the data.
scaler = preprocess_data()

# Load processed data.
data = np.load("data/processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Training for RNN model
print("📌 Training RNN model...")
rnn_model = build_rnn(seq_length=X_train.shape[1], input_dim=X_train.shape[2])

# Early stopping to prevent overfitting.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

rnn_history = rnn_model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

os.makedirs("saved_models", exist_ok=True)
rnn_model.save("saved_models/rnn_model.h5")
print("✅ RNN model saved to 'saved_models/rnn_model.h5'")

# Training for LSTM model
print("📌 Training LSTM model...")
lstm_model = build_lstm(seq_length=X_train.shape[1], input_dim=X_train.shape[2])

lstm_history = lstm_model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

lstm_model.save("saved_models/lstm_model.h5")
print("✅ LSTM model saved to 'saved_models/lstm_model.h5'")
