
import pandas as pd
import numpy as np

df = pd.read_csv(
    "results/tables/hybrid_signal_diagnostics.csv"
)

TRUE_DRIFT_POINT = 1000
THRESHOLD = 0.30

# Reference confidence from the initial stable period
reference = df[df["record"].between(200, 399)]

reference_confidence = reference["prediction_confidence"].mean()

# ------------------------------------------------
# 1. Two-signal detector
# ------------------------------------------------

reference_error = reference["error"].mean()

df["error_signal"] = (
    (df["rolling_error"] - reference_error).clip(lower=0) / 0.5
).clip(0, 1)

df["relationship_signal"] = (
    df["relationship_change"] / 1.2
).clip(0, 1)

df["two_signal_score"] = (
    0.5 * df["error_signal"]
    + 0.5 * df["relationship_signal"]
)

# ------------------------------------------------
# 2. Three-signal detector
# ------------------------------------------------

# Measure absolute change in rolling confidence.
df["confidence_change"] = (
    df["rolling_confidence"] - reference_confidence
).abs()

df["confidence_signal"] = (
    df["confidence_change"] / 0.20
).clip(0, 1)

df["three_signal_score"] = (
    0.40 * df["error_signal"]
    + 0.25 * df["confidence_signal"]
    + 0.35 * df["relationship_signal"]
)

# ------------------------------------------------
# 3. Evaluate both variants
# ------------------------------------------------

def evaluate(score_column, variant_name):

    alarms = df.loc[
        df[score_column] >= THRESHOLD,
        "record"
    ].tolist()

    false_alarms = [
        point for point in alarms
        if point < TRUE_DRIFT_POINT
    ]

    valid_detections = [
        point for point in alarms
        if point >= TRUE_DRIFT_POINT
    ]

    print(f"\n========== {variant_name} ==========")
    print("False alarm records:", false_alarms[:10])
    print("False alarm count:", len(false_alarms))

    if valid_detections:
        first_detection = valid_detections[0]
        delay = first_detection - TRUE_DRIFT_POINT

        print("First detection:", first_detection)
        print("Detection delay:", delay)
    else:
        print("No drift detected.")

evaluate("two_signal_score", "TWO-SIGNAL DETECTOR")
evaluate("three_signal_score", "THREE-SIGNAL DETECTOR")

# ------------------------------------------------
# 4. Save comparison
# ------------------------------------------------

df.to_csv(
    "results/tables/hybrid_variant_comparison.csv",
    index=False
)

print("\nVariant comparison saved successfully!")
