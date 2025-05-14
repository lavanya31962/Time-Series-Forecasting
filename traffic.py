import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler




# Load raw dataset
data = pd.read_csv('C:/Users/ldontheboina/deeplearning/traffic/traffic.csv')


# Drop timestamp column
data = data.drop(data.columns[0], axis=1)

# Downsample: average every 3 time steps
data_distilled = data.groupby(np.arange(len(data)) // 3).mean()

# Smooth with moving average
data_distilled = data_distilled.rolling(window=5).mean().dropna()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_distilled.values)



# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(scaled_data, sequence_length)

# Now you can build and train your LSTM!
