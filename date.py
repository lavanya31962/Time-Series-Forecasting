


import pandas as pd

# Load the dataset
file_path = 'C:/Users/ldontheboina/deeplearning/traffic/traffic.csv'  
data = pd.read_csv(file_path)

# Drop the timestamp column (usually the first column)
if 'timestamp' in data.columns:
    data = data.drop(columns=['timestamp'])
elif 'date' in data.columns:
    data = data.drop(columns=['date'])
else:
    # If no column name, drop the first column by index
    data = data.drop(data.columns[0], axis=1)

# Now data is numeric!
print(data.head())



