# Time Series Forecasting using LSTM

This repository contains Python scripts and models for time series forecasting using Long Short-Term Memory (LSTM) neural networks. The primary focus is on both synthetic and traffic-related time series data. The project explores model training, prediction, and performance comparisons, including the use of model distillation techniques.

### Scripts
- `date.py`  
  Utilities related to time and date handling.

- `synthetic.py`  
  Handles generation, training, and evaluation of synthetic time series data using LSTM.

- `traffic.py`  
  Builds and evaluates LSTM models on traffic dataset.

- `traffic_distillation.py`  
  Applies knowledge distillation to simplify models trained on traffic data.

- `predict_compare_traffic_models.py`  
  Script for comparing predictions from different traffic models.

### Pre-trained Models

- `synthetic_lstm_model.keras`  
  Pre-trained LSTM model on synthetic data.

- `distilled_synthetic_electricity_lstm_model.keras`  
  Distilled version of the LSTM model for electricity-related synthetic data.

- `lstm_traffic_model.h5`  
  Trained LSTM model for traffic data.

## 📦 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib

Install the dependencies using pip: --pip version (ANYTHING LATEST)

## Execution ---
First execute python data.py
Second python traffic.py
Third execute python synthetic.py
4--> python traffic_distillation.py
5 --> python predict_compare_traffic_models.py

## Make sure the necessary datasets are in place (paths may need to be adjusted based on your environment).




