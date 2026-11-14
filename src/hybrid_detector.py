
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load diagnostic data
df = pd.read_csv(
    "results/tables/hybrid_signal_diagnostics.csv"
)

TRUE_DRIFT_POINT = 1000

# Initial configuration
ERROR_WEIGHT = 0.5
RELATIONSHIP_WEIGHT = 0.5

# -------------------------------------------------
# 1. Calculate reference error rate
# -------------------------------------------------

reference = df[df["record"].between(200, 399)]

reference_error = reference["error"].mean()

print("Reference error rate:", round(reference_error, 4))

# -------------------------------------------------
# 2. Calculate error change
# -------------------------------------------------

# Positive increase over reference error
df["error_change"] = (
    df["rolling_error"] - reference_error
).clip(lower=0)

# Normalize to [0, 1]
df["error_signal"] = (
    df["error_change"] / 0.5
).clip(0, 1)

# -------------------------------------------------
# 3. Normalize relationship change
# -------------------------------------------------

# A provisional scaling factor.
# This will be calibrated using validation streams later.
df["relationship_signal"] = (
    df["relationship_change"] / 1.2
).clip(0, 1)

# -------------------------------------------------
# 4. Calculate hybrid score
# -------------------------------------------------

df["hybrid_score"] = (
    ERROR_WEIGHT * df["error_signal"]
    + RELATIONSHIP_WEIGHT * df["relationship_signal"]
)

# -------------------------------------------------
# 5. Initial threshold-based detection
# -------------------------------------------------

THRESHOLD = 0.5

df["hybrid_alarm"] = (
    df["hybrid_score"] >= THRESHOLD
)

# First alarm after true drift
valid_alarms = df.loc[
    (df["record"] >= TRUE_DRIFT_POINT)
    & df["hybrid_alarm"],
    "record"
].tolist()

false_alarms = df.loc[
    (df["record"] < TRUE_DRIFT_POINT)
    & df["hybrid_alarm"],
    "record"
].tolist()

print("\n========== DRIFTGUARD INITIAL RESULTS ==========")
print("Actual drift point:", TRUE_DRIFT_POINT)
print("Threshold:", THRESHOLD)
print("False alarm records:", false_alarms[:20])
print("Number of false alarm records:", len(false_alarms))

print("Valid alarm records:", valid_alarms[:20])

if valid_alarms:
    first_detection = valid_alarms[0]
    detection_delay = first_detection - TRUE_DRIFT_POINT

    print("First detection:", first_detection)
    print("Detection delay:", detection_delay)
else:
    print("No drift detected.")

# -------------------------------------------------
# 6. Visualize hybrid score
# -------------------------------------------------

plt.figure(figsize=(13, 6))

plt.plot(
    df["record"],
    df["hybrid_score"],
    label="DriftGuard Hybrid Score"
)

plt.axhline(
    THRESHOLD,
    color="orange",
    linestyle="--",
    label="Detection Threshold"
)

plt.axvline(
    TRUE_DRIFT_POINT,
    color="red",
    linestyle="--",
    label="Actual Drift"
)

plt.xlabel("Record Number")
plt.ylabel("Hybrid Drift Score")
plt.title("DriftGuard: Initial Hybrid Drift Detection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/figures/hybrid_detector.png",
    dpi=300
)

plt.show()

# -------------------------------------------------
# 7. Save results
# -------------------------------------------------

df.to_csv(
    "results/tables/hybrid_detector_results.csv",
    index=False
)

print("\nHybrid detector results saved successfully!")
