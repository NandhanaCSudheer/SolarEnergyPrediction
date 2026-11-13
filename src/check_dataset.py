import pandas as pd

# Load the two Plant 1 datasets
generation = pd.read_csv("data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

print("\n========== GENERATION DATA ==========")
print(generation.head())

print("\nGeneration shape:")
print(generation.shape)

print("\nGeneration columns:")
print(generation.columns.tolist())

print("\n========== WEATHER DATA ==========")
print(weather.head())

print("\nWeather shape:")
print(weather.shape)

print("\nWeather columns:")
print(weather.columns.tolist())

print("\n========== MISSING VALUES ==========")

print("\nGeneration missing values:")
print(generation.isnull().sum())

print("\nWeather missing values:")
print(weather.isnull().sum())