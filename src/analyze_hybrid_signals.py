
import pandas as pd
import numpy as np

# Load baseline predictions
df = pd.read_csv("results/tables/baseline_predictions.csv")

DRIFT_POINT = 1000
WINDOW_SIZE = 100

# Calculate prediction error
df["error"] = (df["actual"] != df["predicted"]).astype(int)

# Signal 1: Rolling error rate
df["rolling_error"] = (
    df["error"]
    .rolling(window=WINDOW_SIZE)
    .mean()
)

# Signal 2: Rolling prediction confidence
df["rolling_confidence"] = (
    df["prediction_confidence"]
    .rolling(window=WINDOW_SIZE)
    .mean()
)

# Signal 3: Rolling feature-target relationships
df["rolling_corr_f1"] = (
    df["feature_1"]
    .rolling(window=WINDOW_SIZE)
    .corr(df["actual"])
)

df["rolling_corr_f2"] = (
    df["feature_2"]
    .rolling(window=WINDOW_SIZE)
    .corr(df["actual"])
)

# Establish a reference using the first 200 monitored records.
# These correspond to records 200–399, before the true drift.
reference = df[df["record"].between(200, 399)]

reference_corr_f1 = reference["feature_1"].corr(
    reference["actual"]
)

reference_corr_f2 = reference["feature_2"].corr(
    reference["actual"]
)

print("\n========== REFERENCE RELATIONSHIPS ==========")
print(f"Feature 1 correlation: {reference_corr_f1:.4f}")
print(f"Feature 2 correlation: {reference_corr_f2:.4f}")

# Measure how much the current relationships differ
# from the reference relationships.
df["relationship_change"] = np.sqrt(
    (df["rolling_corr_f1"] - reference_corr_f1) ** 2
    + (df["rolling_corr_f2"] - reference_corr_f2) ** 2
)

# Inspect selected records
checkpoints = [900, 999, 1000, 1025, 1050, 1100, 1200, 1500, 1999]

print("\n========== HYBRID SIGNAL DIAGNOSTICS ==========")
print("True drift point:", DRIFT_POINT)

for record in checkpoints:
    row = df[df["record"] == record]

    if not row.empty:
        row = row.iloc[0]

        print(f"\nRecord: {record}")
        print(f"Rolling error rate: {row['rolling_error']:.4f}")
        print(f"Rolling confidence: {row['rolling_confidence']:.4f}")
        print(f"Feature 1 correlation: {row['rolling_corr_f1']:.4f}")
        print(f"Feature 2 correlation: {row['rolling_corr_f2']:.4f}")
        print(f"Relationship change: {row['relationship_change']:.4f}")

# Save diagnostic results
df.to_csv(
    "results/tables/hybrid_signal_diagnostics.csv",
    index=False
)

print("\nDiagnostic results saved successfully!")
print("Output: results/tables/hybrid_signal_diagnostics.csv")
