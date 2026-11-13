import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/cleaned_solar_data.csv")

print("Dataset shape:", df.shape)

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# 2. AC POWER DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(df["AC_POWER"], bins=50, kde=True)

plt.title("Distribution of AC Power Generation")
plt.xlabel("AC Power")
plt.ylabel("Frequency")

plt.show()


# ==========================================
# 3. IRRADIATION VS AC POWER
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["IRRADIATION"],
    df["AC_POWER"],
    alpha=0.3
)

plt.title("Solar Irradiation vs AC Power")
plt.xlabel("Irradiation")
plt.ylabel("AC Power")

plt.show()


# ==========================================
# 4. TEMPERATURE VS AC POWER
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["MODULE_TEMPERATURE"],
    df["AC_POWER"],
    alpha=0.3
)

plt.title("Module Temperature vs AC Power")
plt.xlabel("Module Temperature")
plt.ylabel("AC Power")

plt.show()


# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(12, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()