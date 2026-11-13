import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


# Load cleaned dataset
df = pd.read_csv("data/cleaned_solar_data.csv")

# Separate features and target
X = df.drop("AC_POWER", axis=1)
y = df["AC_POWER"]


# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train model
print("Training Random Forest...")

model.fit(X, y)

print("Training completed.")


# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")
print("Location: models/random_forest_model.pkl")