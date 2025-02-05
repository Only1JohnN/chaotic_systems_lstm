# type: ignore tensorflow

import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import load_model

def plot_results(true_values, predicted_values, filename="results/plot.png"):
    """
    Plots true vs predicted values (for the x-component as an example) and saves the figure.
    :param true_values: Array of true values.
    :param predicted_values: Array of predicted values.
    :param filename: File path to save the plot.
    """
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values[:, 0], label="True x")
    plt.plot(predicted_values[:, 0], label="Predicted x", linestyle="--")
    plt.title("True vs Predicted x values")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved to '{filename}'")

def visualize_model(model_path, data_path="data/processed_data.npz"):
    """
    Loads a saved model and processed test data, generates predictions,
    and plots the true vs predicted values.
    :param model_path: Path to the saved model file.
    :param data_path: Path to the processed data (.npz file).
    """
    # Check if processed data exists
    if not os.path.exists(data_path):
        print(f"❌ Processed data file not found at {data_path}.")
        return

    # Load processed test data.
    data = np.load(data_path)
    if "X_test" not in data or "y_test" not in data:
        print("❌ Processed data does not contain 'X_test' and 'y_test'.")
        return

    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"Loaded test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train the model first.")
        return

    # Load the saved model.
    try:
        model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate predictions.
    try:
        predicted = model.predict(X_test)
        print("Prediction completed successfully.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # Verify that predictions have the correct shape.
    print(f"Predictions shape: {predicted.shape}")

    # Create a filename based on the model name.
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    filename = f"results/{model_name}_predictions.png"
    
    # Plot and save the results.
    plot_results(y_test, predicted, filename=filename)

if __name__ == "__main__":
    # Example: Visualize using the LSTM model.
    model_path = "saved_models/lstm_model.h5"  # You can change this to "rnn_model.h5" for the RNN model
    visualize_model(model_path)
