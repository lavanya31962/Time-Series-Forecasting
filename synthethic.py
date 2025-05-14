import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
days = 30
hours_per_day = 24
total_hours = days * hours_per_day

# Create time index
timestamps = pd.date_range(start='2024-01-01', periods=total_hours, freq='H')

# Generate synthetic traffic:
# Daily sinusoidal pattern + weekly modulation + noise
base_traffic = (
    100 +  # baseline
    50 * np.sin(2 * np.pi * np.arange(total_hours) / 24) +  # daily cycle
    20 * np.sin(2 * np.pi * np.arange(total_hours) / (24 * 7)) +  # weekly pattern
    10 * np.random.randn(total_hours)  # random noise
)

# Ensure no negative traffic values
base_traffic = np.maximum(base_traffic, 0)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'traffic': base_traffic
})

# Save to CSV
df.to_csv('synthetic_traffic.csv', index=False)
print("✅ Synthetic traffic data saved as 'synthetic_traffic.csv'")

# Show first 48 rows in table format
print("\n📋 First 48 rows of synthetic traffic data:")
print(df.head(48).to_string(index=False))

# Plot sample
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['traffic'])
plt.title('Synthetic Traffic Time Series')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.tight_layout()
plt.show()