import pandas as pd
import matplotlib.pyplot as plt
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

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Print results
print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()