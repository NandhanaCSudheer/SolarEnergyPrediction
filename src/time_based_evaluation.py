import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# Load original generation and weather data
# ==========================================

generation = pd.read_csv(
    "data/Plant_1_Generation_Data.csv"
)

weather = pd.read_csv(
    "data/Plant_1_Weather_Sensor_Data.csv"
)


# Convert DATE_TIME to datetime
generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"],
    dayfirst=False
)


# ==========================================
# Merge generation and weather data
# ==========================================

df = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)


# ==========================================
# Create time features
# ==========================================

df["HOUR"] = df["DATE_TIME"].dt.hour
df["MINUTE"] = df["DATE_TIME"].dt.minute
df["DAY"] = df["DATE_TIME"].dt.day
df["MONTH"] = df["DATE_TIME"].dt.month


# ==========================================
# Select required features
# ==========================================

features = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "HOUR",
    "MINUTE",
    "DAY",
    "MONTH"
]

target = "AC_POWER"


df = df[features + [target, "DATE_TIME"]]

# Remove missing values
df = df.dropna()

# Sort chronologically
df = df.sort_values("DATE_TIME").reset_index(drop=True)


# ==========================================
# Time-based train/test split
# ==========================================

split_index = int(len(df) * 0.8)

train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]


X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


print("========== TIME-BASED SPLIT ==========")

print("Total samples:", len(df))
print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

print("\nTraining period:")
print(train_data["DATE_TIME"].min())
print("to")
print(train_data["DATE_TIME"].max())

print("\nTesting period:")
print(test_data["DATE_TIME"].min())
print("to")
print(test_data["DATE_TIME"].max())


# ==========================================
# Linear Regression
# ==========================================

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression completed.")


# ==========================================
# Random Forest
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest completed.")


# ==========================================
# Results
# ==========================================

print("\n========== TIME-BASED RESULTS ==========")

print("\nMetric              Linear Regression      Random Forest")

print(
    f"MAE                 {linear_mae:.4f}                {rf_mae:.4f}"
)

print(
    f"MSE                 {linear_mse:.4f}             {rf_mse:.4f}"
)

print(
    f"RMSE                {linear_rmse:.4f}                {rf_rmse:.4f}"
)

print(
    f"R²                  {linear_r2:.4f}                  {rf_r2:.4f}"
)