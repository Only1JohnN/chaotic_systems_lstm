import os
import sys
import subprocess

# Create directories for logs and results if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define the log file path
log_file_path = os.path.join("logs", "run_log.txt")

# List of scripts to run in sequence
scripts = [
    "src/lorenz.py",  # This will generate Lorenz data if it doesn't exist
    "src/data_preprocessing.py",  # This will preprocess the data
    "src/train.py",  # This will train both models (LSTM and RNN)
    "src/visualize.py"  # This will visualize the results
]

with open(log_file_path, "w") as log_file:
    for script in scripts:
        log_file.write(f"\n📌 Running {script}...\n")
        print(f"📌 Running {script}...")
        # Run the script and capture both stdout and stderr into the log file
        result = subprocess.run(["python", script], stdout=log_file, stderr=log_file)
        if result.returncode != 0:
            log_file.write(f"❌ Error encountered when running {script}.\n")
            print(f"❌ Error encountered when running {script}. Check the log for details.")
        else:
            log_file.write(f"✅ Finished running {script}.\n")
            print(f"✅ Finished running {script}.")

print(f"\n✅ All scripts executed. Check '{log_file_path}' for logs and the 'results' directory for output plots.")
