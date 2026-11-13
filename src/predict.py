import joblib
import pandas as pd


# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

print("Random Forest model loaded successfully!")


# Get input values from user
ambient_temperature = float(
    input("Enter ambient temperature: ")
)

module_temperature = float(
    input("Enter module temperature: ")
)

irradiation = float(
    input("Enter irradiation: ")
)

hour = int(
    input("Enter hour (0-23): ")
)

minute = int(
    input("Enter minute (0-59): ")
)

day = int(
    input("Enter day of month (1-31): ")
)

month = int(
    input("Enter month (1-12): ")
)


# Create input DataFrame
input_data = pd.DataFrame([{
    "AMBIENT_TEMPERATURE": ambient_temperature,
    "MODULE_TEMPERATURE": module_temperature,
    "IRRADIATION": irradiation,
    "HOUR": hour,
    "MINUTE": minute,
    "DAY": day,
    "MONTH": month
}])


# Make prediction
prediction = model.predict(input_data)


# Display result
print("\n========== PREDICTION ==========")
print("Predicted AC Power:", prediction[0])