import pandas as pd

# Load a CSV dataset
data = pd.read_csv('Life_Expectancy_Data.csv')

# Display the first 5 rows of the dataset
print(data.head())

# Display basic statistics of numerical columns
print(data.describe())
