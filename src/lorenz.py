import numpy as np
from scipy.integrate import solve_ivp
import os
import pandas as pd

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span=(0, 40), t_eval_step=0.01, init_state=[1, 1, 1]):
    """
    Generates Lorenz system data using solve_ivp.
    :param t_span: Tuple specifying the time range.
    :param t_eval_step: Time resolution.
    :param init_state: Initial conditions.
    :return: time vector and data (array of shape [time_steps, 3]).
    """
    t_eval = np.arange(t_span[0], t_span[1], t_eval_step)
    sol = solve_ivp(lorenz, t_span, init_state, t_eval=t_eval)
    return sol.t, sol.y.T  # Transpose so that each row is a timestep

if __name__ == "__main__":
    print("📌 Generating Lorenz data...")
    os.makedirs("data", exist_ok=True)
    t, data = generate_lorenz_data()
    df = pd.DataFrame(data, columns=["x", "y", "z"])
    df.to_csv("data/lorenz_data.csv", index=False)
    print("✅ Lorenz data saved to 'data/lorenz_data.csv'")
