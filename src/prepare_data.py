import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

generation = pd.read_csv("data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")


# ==========================================
# 2. CONVERT DATE_TIME
# ==========================================

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"]
)


# ==========================================
# 3. MERGE DATASETS
# ==========================================

df = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="left"
)


# ==========================================
# 4. REMOVE MISSING VALUES
# ==========================================

df = df.dropna()


# ==========================================
# 5. CREATE TIME FEATURES
# ==========================================

df["HOUR"] = df["DATE_TIME"].dt.hour
df["MINUTE"] = df["DATE_TIME"].dt.minute
df["DAY"] = df["DATE_TIME"].dt.day
df["MONTH"] = df["DATE_TIME"].dt.month


# ==========================================
# 6. SELECT FEATURES
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

df = df[features + [target]]


# ==========================================
# 7. SAVE CLEAN DATA
# ==========================================

df.to_csv("data/cleaned_solar_data.csv", index=False)


# ==========================================
# 8. DISPLAY INFORMATION
# ==========================================

print("\n========== CLEANED DATA ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nCleaned dataset saved as:")
print("data/cleaned_solar_data.csv")