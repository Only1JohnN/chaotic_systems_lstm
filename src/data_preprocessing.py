import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(seq_length=10, test_size=2000):
    """
    Loads Lorenz data, scales it, and creates sliding windows.
    :param seq_length: Length of input sequences.
    :param test_size: Number of samples reserved for testing.
    :return: Fitted scaler.
    """
    df = pd.read_csv("data/lorenz_data.csv")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data: last 'test_size' samples for testing.
    if test_size < len(X):
        X_train = X[:-test_size]
        y_train = y[:-test_size]
        X_test = X[-test_size:]
        y_test = y[-test_size:]
    else:
        X_train, y_train, X_test, y_test = X, y, X, y

    os.makedirs("data", exist_ok=True)
    np.savez("data/processed_data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    joblib.dump(scaler, "data/scaler.pkl")
    print("✅ Data preprocessing complete. Processed data saved to 'data/processed_data.npz'")
    return scaler

if __name__ == "__main__":
    preprocess_data()
