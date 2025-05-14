import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 1. Load and clean traffic dataset
file_path = 'C:/Users/ldontheboina/deeplearning/traffic/traffic.csv'  
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = pd.read_csv(file_path)

# Drop timestamp (first column)
data = data.drop(data.columns[0], axis=1)

# Fill missing values
data = data.ffill()

# Keep only numeric columns
data = data.select_dtypes(include=['float64', 'int64'])

if data.empty:
    raise ValueError("No numeric columns found after cleaning!")

# 2. Downsample / Aggregate: average every 3 time steps
data_distilled = data.groupby(np.arange(len(data)) // 3).mean()

# 3. Smooth data with moving average
data_distilled = data_distilled.rolling(window=5).mean().dropna()

# 4. Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_distilled.values)

# 5. Create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(scaled_data, sequence_length)

# 6. Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# 7. Build stacked LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    LSTM(32),
    Dense(X.shape[2])
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 8. Train the model
history = model.fit(X_train, y_train, epochs=8
                    , batch_size=32, validation_split=0.1)

# 9. Evaluate model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.6f}')

# 10. Save model
model.save('traffic_lstm_distilled_model.h5')
print("Model saved as traffic_lstm_distilled_model.h5")

# 11. Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.show()
