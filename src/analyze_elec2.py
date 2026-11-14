
import pandas as pd
from river import datasets

# Load dataset in its original order
dataset = datasets.Elec2()

records = []

for index, (features, target) in enumerate(dataset):
    row = features.copy()
    row["target"] = target
    row["record"] = index
    records.append(row)

df = pd.DataFrame(records)

print("\n========== DATASET SHAPE ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string(index=False))

print("\n========== LAST 5 ROWS ==========")
print(df.tail().to_string(index=False))

print("\n========== CLASS DISTRIBUTION ==========")
print(df["target"].value_counts())
print("\nClass percentages:")
print(
    (df["target"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== FEATURE DATA TYPES ==========")
print(df.dtypes)

print("\n========== FEATURE STATISTICS ==========")
print(df.drop(columns=["record"]).describe().round(4))

print("\n========== TEMPORAL INFORMATION ==========")
print("Unique dates:", df["date"].nunique())
print("Unique days:", df["day"].nunique())
print("Unique periods:", df["period"].nunique())

print("\n========== TARGET TRANSITIONS ==========")
changes = (df["target"] != df["target"].shift()).sum()
print("Number of target class changes:", int(changes))

print("\n========== DUPLICATE ROWS ==========")
print(
    "Duplicate feature rows:",
    df.drop(columns=["record", "target"]).duplicated().sum()
)

# Save a local copy for reproducible experiments
output_path = "data/processed/elec2_real_world.csv"

from pathlib import Path
Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("\nSaved dataset to:", output_path)
print("\nElec2 analysis completed.")
