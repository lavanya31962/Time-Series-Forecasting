import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
electricity_df = pd.read_csv("C:/Users/dhath/Downloads/all_six_datasets/all_six_datasets/electricity/electricity.csv")
weather_df = pd.read_csv("C:/Users/dhath/Downloads/all_six_datasets/all_six_datasets/weather/weather.csv")

# Convert 'date' columns to datetime format
electricity_df['date'] = pd.to_datetime(electricity_df['date'])
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Merge datasets on 'date'
merged_df = pd.merge(electricity_df, weather_df, on='date', how='inner')

# Handle missing values (fill with column mean)
merged_df.fillna(merged_df.mean(), inplace=True)

# Set 'date' as index
merged_df.set_index('date', inplace=True)

# Summary statistics
print("Summary Statistics:\n", merged_df.describe())

# Time-series plot of total electricity consumption and temperature
plt.figure(figsize=(12, 6))
plt.plot(merged_df.index, merged_df.iloc[:, 1:321].sum(axis=1), label="Total Electricity Consumption", color='b')
plt.ylabel("Electricity Consumption")

plt.twinx()
plt.plot(merged_df.index, merged_df['T (degC)'], label="Temperature (°C)", color='r', alpha=0.6)
plt.ylabel("Temperature (°C)")

plt.title("Electricity Consumption vs Temperature")
plt.legend(loc="upper left")
plt.show()

# Save cleaned dataset
merged_df.to_csv("/mnt/data/cleaned_data.csv")

print("Cleaned dataset saved as 'cleaned_data.csv'")
