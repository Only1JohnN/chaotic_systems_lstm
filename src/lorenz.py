import numpy as np
from scipy.integrate import odeint
import pandas as pd
import os

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(time_steps=10000, dt=0.01, init_state=[1,1,1]):
    t = np.linspace(0, time_steps*dt, time_steps)
    solution = odeint(lorenz, init_state, t)
    return t, solution

if __name__ == "__main__":
    print("\n📌 Generating Lorenz chaotic data...")
    os.makedirs("data", exist_ok=True)
    t, data = generate_lorenz_data()
    pd.DataFrame(data, columns=["x", "y", "z"]).to_csv("data/lorenz_data.csv", index=False)
    print("✅ Lorenz data saved to 'data/lorenz_data.csv'")
