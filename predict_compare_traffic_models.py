import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 1. Load real and synthetic traffic datasets
real_data_path = 'C:/Users/ldontheboina/deeplearning/traffic/traffic.csv'
synthetic_data_path = 'synthetic_traffic.csv'

if not os.path.exists(real_data_path):
    raise FileNotFoundError(f"Real traffic file not found: {real_data_path}")
if not os.path.exists(synthetic_data_path):
    raise FileNotFoundError(f"Synthetic traffic file not found: {synthetic_data_path}")

real_data = pd.read_csv(real_data_path)
synthetic_data = pd.read_csv(synthetic_data_path)

# 2. Drop timestamp columns and select only the first numeric column for real and synthetic data
real_data = real_data.drop(columns=[real_data.columns[0]])
synthetic_data = synthetic_data.drop(columns=[synthetic_data.columns[0]])

real_data = real_data.ffill()

# Automatically detect the first numeric column
numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in real traffic data!")
real_data = real_data[[numeric_cols[0]]]

# 3. Preprocessing
real_data = real_data.groupby(np.arange(len(real_data)) // 3).mean()
real_data = real_data.rolling(window=5).mean().dropna()

synthetic_data = synthetic_data.groupby(np.arange(len(synthetic_data)) // 3).mean()
synthetic_data = synthetic_data.rolling(window=5).mean().dropna()

# 4. Normalize
scaler_real = MinMaxScaler()
real_scaled = scaler_real.fit_transform(real_data.values)

scaler_synth = MinMaxScaler()
synthetic_scaled = scaler_synth.fit_transform(synthetic_data.values)

# 5. Create sequences
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 24
X_real, y_real = create_sequences(real_scaled, sequence_length)
X_synth, y_synth = create_sequences(synthetic_scaled, sequence_length)

# 6. Train/test split
split_real = int(0.8 * len(X_real))
X_real_train, X_real_test = X_real[:split_real], X_real[split_real:]
y_real_train, y_real_test = y_real[:split_real], y_real[split_real:]

split_synth = int(0.8 * len(X_synth))
X_synth_train, X_synth_test = X_synth[:split_synth], X_synth[split_synth:]
y_synth_train, y_synth_test = y_synth[:split_synth], y_synth[split_synth:]

# 7. Build model function
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 8. Train models
model_real = build_model((sequence_length, X_real.shape[2]))
model_synth = build_model((sequence_length, X_synth.shape[2]))

model_real.fit(X_real_train, y_real_train, epochs=20, batch_size=32, validation_split=0.1)
model_synth.fit(X_synth_train, y_synth_train, epochs=20, batch_size=32, validation_split=0.1)

# 9. Predict
y_real_pred = model_real.predict(X_real_test)
y_synth_pred = model_synth.predict(X_synth_test)

# 10. Inverse Transform
y_real_test_inv = scaler_real.inverse_transform(y_real_test)
y_real_pred_inv = scaler_real.inverse_transform(y_real_pred)

y_synth_test_inv = scaler_synth.inverse_transform(y_synth_test)
y_synth_pred_inv = scaler_synth.inverse_transform(y_synth_pred)

# 11. Metrics
mse_real = mean_squared_error(y_real_test_inv, y_real_pred_inv)
mae_real = mean_absolute_error(y_real_test_inv, y_real_pred_inv)

mse_synth = mean_squared_error(y_synth_test_inv, y_synth_pred_inv)
mae_synth = mean_absolute_error(y_synth_test_inv, y_synth_pred_inv)

print(f"Real Model - MSE: {mse_real:.3f}, MAE: {mae_real:.3f}")
print(f"Synthetic Model - MSE: {mse_synth:.3f}, MAE: {mae_synth:.3f}")

# 12. Plot predictions
plt.figure(figsize=(12,6))
plt.plot(y_real_test_inv, label='Real Traffic')
plt.plot(y_real_pred_inv, label='Real Model Prediction')
plt.title('Real Traffic: Real vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_synth_test_inv, label='Synthetic Traffic')
plt.plot(y_synth_pred_inv, label='Synthetic Model Prediction')
plt.title('Synthetic Traffic: Real vs Predicted')
plt.xlabel('Time Step')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()