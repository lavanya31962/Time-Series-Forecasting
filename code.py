import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/dhath/Downloads/all_six_datasets/all_six_datasets/electricity/electricity.csv"
df = pd.read_csv(file_path)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set 'date' as index
df.set_index('date', inplace=True)

# Handle missing values (fill with mean)
df.fillna(df.mean(), inplace=True)

# Summary statistics
print(df.describe())

# Time-series plot for overall electricity consumption (sum of all columns)
plt.figure(figsize=(12, 6))
df.sum(axis=1).plot(title="Total Electricity Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Electricity Consumption")
plt.grid()
plt.show()
