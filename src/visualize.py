import matplotlib.pyplot as plt
import os

def plot_results(true_values, predicted_values, filename="results/plot.png"):
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_values[:, 0], label="Actual X")
    plt.plot(predicted_values[:, 0], label="Predicted X", linestyle="dashed")
    plt.legend()
    plt.title("Actual vs Predicted X values")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print("✅ Visualization module loaded")
