import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
traffic_file_path = "C:/Users/dhath/Downloads/all_six_datasets/all_six_datasets/traffic/traffic.csv"  # Update with your file path
traffic_df = pd.read_csv(traffic_file_path)

# Convert 'date' to datetime
traffic_df['date'] = pd.to_datetime(traffic_df['date'])

# Set 'date' as the index
traffic_df.set_index('date', inplace=True)

# Handle missing values (fill with column mean)
traffic_df.fillna(traffic_df.mean(), inplace=True)

# Summary statistics
print("Summary Statistics:\n", traffic_df.describe())

# Time-series plot of total traffic
plt.figure(figsize=(12, 6))
traffic_df.iloc[:, :-1].sum(axis=1).plot(title="Total Traffic Flow Over Time", color='b')
plt.xlabel("Date")
plt.ylabel("Traffic Flow")
plt.grid()
plt.show()

# Save cleaned dataset
traffic_df.to_csv("cleaned_traffic_data.csv")
print("Cleaned dataset saved as 'cleaned_traffic_data.csv'")
