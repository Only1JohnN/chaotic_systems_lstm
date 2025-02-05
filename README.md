# Chaotic Systems-LSTM

This project aims to learn and model chaotic systems using Long Short-Term Memory (LSTM) networks and Recurrent Neural Networks (RNNs), with a focus on the Lorenz system. The repository includes data generation, preprocessing, model definition, training, and evaluation.

## Project Structure

```
chaos-lstm
│── 📂 src
│   ├── lorenz.py                 # Generates Lorenz system data
│   ├── data_preprocessing.py     # Prepares dataset for training
│   ├── lstm_model.py             # Defines LSTM model
│   ├── rnn_model.py              # Defines RNN model
│   ├── visualize.py              # Handles visualization of predictions
│   ├── train.py                  # Trains both the LSTM and RNN models
│
│── 📂 data                       # Auto-created: Stores dataset  
│── 📂 saved_models               # Auto-created: Stores trained models (LSTM and RNN models)
│── 📂 results                    # Auto-created: Stores plots of predictions (currently only for LSTM)
│── 📂 logs                       # Auto-created: Stores logs from script executions
│
│── 📜 run_all.py                 # 💡 One script to run everything (train and visualize)
│── 📜 requirements.txt           # Dependencies 
│── 📜 README.md                  # Project documentation

```

## Requirements

- Python 3.10
- Install dependencies via:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

To execute the full pipeline, use the `run_all.py` script:
```bash
python run_all.py
```
This script will:
- Generate Lorenz system data (if not already present)
- Preprocess the data
- Train both the LSTM and RNN models
- Save the trained models
- Visualize the results by plotting predictions for the LSTM model

The log of the script execution will be saved in the `logs/run_log.txt` file. The plot of the predictions for the LSTM model will be saved in the `results/` directory.

## Results

Currently, only the **LSTM model predictions** are visualized and saved. The following image illustrates the predictions made by the LSTM model.

### LSTM Model Predictions

![LSTM Predictions](results/lstm_model__predictions.png)

## Model Details

### LSTM Model
The LSTM model is defined in `src/lstm_model.py` and is used to predict the chaotic behavior of the Lorenz system.

### RNN Model
The RNN model is defined in `src/rnn_model.py` and is also applied to model the chaotic behavior. While the RNN model is trained, its predictions are not yet visualized in this version of the project.

Both models are trained using the same dataset and hyperparameters (with early stopping to prevent overfitting) but may yield different results due to their architectural differences.

## Author
Adeniyi John