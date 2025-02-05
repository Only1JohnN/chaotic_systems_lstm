from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import joblib

def preprocess_data(seq_length=10):
    df = pd.read_csv("data/lorenz_data.csv")
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])

    os.makedirs("data", exist_ok=True)
    np.savez("data/processed_data.npz", X_train=X[:-2000], y_train=y[:-2000], X_test=X[-2000:], y_test=y[-2000:])
    
    joblib.dump(scaler, "data/scaler.pkl")  # Save the scaler
    print("✅ Data preprocessing complete. Saved to 'data/processed_data.npz'")
    return scaler

if __name__ == "__main__":
    preprocess_data()
