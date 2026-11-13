import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/cleaned_solar_data.csv")

# Separate features and target
X = df.drop("AC_POWER", axis=1)
y = df["AC_POWER"]


# Same train-test split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# Linear Regression
# ==============================

print("Training Linear Regression...")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression completed.")


# ==============================
# Random Forest
# ==============================

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


# ==============================
# Print comparison
# ==============================

print("\n========== MODEL COMPARISON ==========")

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


# ==============================
# Create comparison graph
# ==============================

metrics = ["MAE", "MSE", "RMSE", "R²"]

linear_values = [
    linear_mae,
    linear_mse,
    linear_rmse,
    linear_r2
]

rf_values = [
    rf_mae,
    rf_mse,
    rf_rmse,
    rf_r2
]


x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    linear_values,
    width,
    label="Linear Regression"
)

plt.bar(
    x + width / 2,
    rf_values,
    width,
    label="Random Forest"
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Metric Value")
plt.title("Linear Regression vs Random Forest")
plt.xticks(x, metrics)
plt.legend()

plt.tight_layout()
plt.show()