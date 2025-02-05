import os
import numpy as np
import tensorflow as tf
import joblib
from src.lorenz import generate_lorenz_data
from src.data_preprocessing import preprocess_data
from src.lstm_model import build_lstm
from src.rnn_model import build_rnn
from src.visualize import plot_results

### STEP 1: GENERATE DATA ###
print("\n📌 Generating Lorenz data...")
os.makedirs("data", exist_ok=True)  

if not os.path.exists("data/lorenz_data.csv"):
    t, data = generate_lorenz_data()
    np.savetxt("data/lorenz_data.csv", data, delimiter=",", header="x,y,z", comments="")
    print("✅ Lorenz data saved to 'data/lorenz_data.csv'")

### STEP 2: PREPROCESS DATA ###
print("\n📌 Preprocessing data...")
if not os.path.exists("data/scaler.pkl"):
    preprocess_data()

# Load processed data
data = np.load("data/processed_data.npz")
X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
scaler = joblib.load("data/scaler.pkl")

### STEP 3: TRAIN MODELS ###
os.makedirs("saved_models", exist_ok=True)

# Train LSTM
print("\n📌 Training LSTM model...")
lstm = build_lstm()
lstm.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
lstm.save("saved_models/lstm_model.keras")  # 🔹 Fix: Use .keras format

# Train RNN
print("\n📌 Training RNN model...")
rnn = build_rnn()
rnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
rnn.save("saved_models/rnn_model.keras")  # 🔹 Fix: Use .keras format

### STEP 4: EVALUATE & VISUALIZE ###
print("\n📌 Evaluating models...")
os.makedirs("results", exist_ok=True)

# 🔹 Fix: Ensure loss function is explicitly set when loading
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load trained models
lstm = tf.keras.models.load_model("saved_models/lstm_model.keras", custom_objects=custom_objects)
rnn = tf.keras.models.load_model("saved_models/rnn_model.keras", custom_objects=custom_objects)

# Make predictions
lstm_pred = lstm.predict(X_test)
rnn_pred = rnn.predict(X_test)

# Save results
plot_results(y_test, lstm_pred, "results/lstm_predictions.png")
plot_results(y_test, rnn_pred, "results/rnn_predictions.png")

print("\n✅ All steps completed successfully! Check 'results/' for outputs.")
