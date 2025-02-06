import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    """Defines the Lorenz system differential equations."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span=(0, 40), t_eval_step=0.01, init_state=[1, 1, 1]):
    """Generates Lorenz system data and returns time values and corresponding x, y, z states."""
    print("📌 Generating Lorenz system data...")
    t_eval = np.arange(t_span[0], t_span[1], t_eval_step)
    sol = solve_ivp(lorenz, t_span, init_state, t_eval=t_eval)
    print("✅ Data generation complete.")
    return sol.t, sol.y.T

def save_data_to_csv(data, filename="data/lorenz_data.csv"):
    """Saves Lorenz data to CSV file."""
    os.makedirs("data", exist_ok=True)
    print(f"📌 Saving Lorenz data to '{filename}'...")
    pd.DataFrame(data, columns=["x", "y", "z"]).to_csv(filename, index=False)
    print(f"✅ Data saved successfully at '{filename}'.")

def plot_lorenz_attractor(data, filename="results/lorenz_attractor.png"):
    """Plots the Lorenz attractor and saves the image."""
    os.makedirs("results", exist_ok=True)
    print("📌 Generating Lorenz attractor visualization...")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2], color='b', alpha=0.7)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor (Chaotic System)")

    plt.savefig(filename)
    plt.show()
    print(f"✅ Lorenz attractor visualization saved at '{filename}'.")

if __name__ == "__main__":
    print("🔹 Running Lorenz system script...")
    
    # Generate data
    t, data = generate_lorenz_data()
    
    # Save data
    save_data_to_csv(data)

    # Generate and save visualization
    plot_lorenz_attractor(data)

    print("🎯 Lorenz system script execution completed.")
