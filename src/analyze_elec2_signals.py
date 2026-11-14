
import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/elec2_real_world.csv")

predictions = pd.read_csv(
    "results/tables/elec2_baseline_predictions.csv"
)

FEATURES = [
    "date",
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer"
]

WINDOW = 200
INITIAL_TRAIN_SIZE = 1000

CALIBRATION_START = 1199
MONITORING_START = 1399

OUTPUT_PATH = "results/tables/elec2_signal_diagnostics.csv"


# ============================================================
# 2. DATA VALIDATION
# ============================================================

required_columns = FEATURES + ["target", "record"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in original dataset: {missing_columns}"
    )

prediction_columns = [
    "record",
    "actual",
    "predicted",
    "correct",
    "prediction_confidence"
]

missing_prediction_columns = [
    col for col in prediction_columns
    if col not in predictions.columns
]

if missing_prediction_columns:
    raise ValueError(
        f"Missing prediction columns: {missing_prediction_columns}"
    )

if df["record"].duplicated().any():
    raise ValueError("Duplicate record IDs in original dataset.")

if predictions["record"].duplicated().any():
    raise ValueError("Duplicate record IDs in predictions.")

if len(df) < MONITORING_START:
    raise ValueError("Dataset is too small for the selected timeline.")


# ============================================================
# 3. REFERENCE FEATURE-TARGET RELATIONSHIPS
# ============================================================

df["target_numeric"] = df["target"].astype(int)

reference_data = df.iloc[:INITIAL_TRAIN_SIZE]

reference_correlations = {}

for feature in FEATURES:

    correlation = reference_data[feature].corr(
        reference_data["target_numeric"]
    )

    reference_correlations[feature] = correlation


print("\n========== REFERENCE CORRELATIONS ==========")

for feature, correlation in reference_correlations.items():

    print(f"{feature}: {correlation}")


# ============================================================
# 4. MERGE BASELINE PREDICTIONS
# ============================================================

evaluation = df.merge(
    predictions[prediction_columns],
    on="record",
    how="inner",
    suffixes=("", "_prediction"),
    validate="one_to_one"
)

evaluation = (
    evaluation
    .sort_values("record")
    .reset_index(drop=True)
)

if len(evaluation) != len(predictions):
    raise ValueError(
        "Some prediction records could not be matched to the dataset."
    )

if not (
    evaluation["target"].astype(bool)
    == evaluation["actual"].astype(bool)
).all():

    raise ValueError(
        "Target mismatch between dataset and predictions."
    )

if not evaluation["correct"].isin([0, 1, True, False]).all():
    raise ValueError("Unexpected values in prediction correctness column.")

if not evaluation["prediction_confidence"].between(0, 1).all():
    raise ValueError("Prediction confidence must be between 0 and 1.")


# ============================================================
# 5. PREDICTION ERROR
# ============================================================

evaluation["prediction_error"] = (
    1 - evaluation["correct"].astype(int)
)


# ============================================================
# 6. ROLLING ERROR AND CONFIDENCE
# ============================================================

evaluation["rolling_error"] = (
    evaluation["prediction_error"]
    .rolling(
        WINDOW,
        min_periods=WINDOW
    )
    .mean()
)

evaluation["rolling_confidence"] = (
    evaluation["prediction_confidence"]
    .rolling(
        WINDOW,
        min_periods=WINDOW
    )
    .mean()
)


# ============================================================
# 7. REFERENCE ERROR AND CONFIDENCE
# ============================================================

calibration_reference = evaluation.iloc[:WINDOW].copy()

if len(calibration_reference) != WINDOW:
    raise ValueError(
        "Insufficient initial prediction records for reference values."
    )

reference_error = (
    calibration_reference["prediction_error"].mean()
)

reference_confidence = (
    calibration_reference["prediction_confidence"].mean()
)

print("\n========== REFERENCE ERROR AND CONFIDENCE ==========")

print("Reference error:", round(reference_error, 4))
print("Reference confidence:", round(reference_confidence, 4))


# ============================================================
# 8. CAUSAL ROLLING FEATURE-TARGET CORRELATIONS
# ============================================================

rolling_correlations = {}

for feature in FEATURES:

    full_rolling_corr = (
        df[feature]
        .rolling(
            WINDOW,
            min_periods=WINDOW
        )
        .corr(df["target_numeric"])
    )

    correlation_series = pd.Series(
        full_rolling_corr.to_numpy(),
        index=df["record"]
    )

    rolling_correlations[feature] = (
        evaluation["record"].map(correlation_series)
    )


# ============================================================
# 9. SELECT VALID REFERENCE FEATURES
# ============================================================

valid_features = [
    feature
    for feature in FEATURES
    if pd.notna(reference_correlations[feature])
]

print("\n========== RELATIONSHIP MONITORING ==========")

print("Features included:")

for feature in valid_features:
    print(feature)

if not valid_features:
    raise ValueError(
        "No valid reference correlations available."
    )


# ============================================================
# 10. CALCULATE RELATIONSHIP CHANGE
# ============================================================

relationship_changes = []

for index in range(len(evaluation)):

    changes = []

    for feature in valid_features:

        current_corr = rolling_correlations[feature].iloc[index]

        reference_corr = reference_correlations[feature]

        if pd.notna(current_corr):

            changes.append(
                abs(current_corr - reference_corr)
            )

    if changes:

        relationship_changes.append(
            np.mean(changes)
        )

    else:

        relationship_changes.append(np.nan)


evaluation["relationship_change"] = relationship_changes


# ============================================================
# 11. RELATIVE SIGNAL CHANGES
# ============================================================

evaluation["error_increase"] = (
    evaluation["rolling_error"] - reference_error
)

evaluation["confidence_drop"] = (
    reference_confidence - evaluation["rolling_confidence"]
)


# ============================================================
# 12. CHRONOLOGICAL PHASE ASSIGNMENT
# ============================================================

evaluation["monitoring_phase"] = np.select(
    [
        evaluation["record"] < CALIBRATION_START,
        evaluation["record"] < MONITORING_START
    ],
    [
        "warmup",
        "calibration"
    ],
    default="monitoring"
)


# ============================================================
# 13. PREPARE OUTPUT
# ============================================================

output_columns = [
    "record",
    "actual",
    "predicted",
    "correct",
    "prediction_confidence",
    "rolling_error",
    "error_increase",
    "rolling_confidence",
    "confidence_drop",
    "relationship_change",
    "monitoring_phase"
]

output = evaluation[output_columns].copy()


# ============================================================
# 14. CALIBRATION VALIDATION
# ============================================================

calibration_data = output[
    output["monitoring_phase"] == "calibration"
].copy()

signal_columns = [
    "rolling_error",
    "confidence_drop",
    "relationship_change"
]

print("\n========== CALIBRATION VALIDATION ==========")

print(
    "Calibration records:",
    calibration_data["record"].min(),
    "to",
    calibration_data["record"].max()
)

print(
    "Calibration observations:",
    len(calibration_data)
)

print("\nValid observations per signal:")

print(
    calibration_data[signal_columns].count()
)

print("\nCalibration signal statistics:")

print(
    calibration_data[signal_columns]
    .describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )
    .round(4)
    .to_string()
)

if len(calibration_data) != 200:

    raise ValueError(
        "Expected exactly 200 calibration observations."
    )

if calibration_data[signal_columns].isna().any().any():

    raise ValueError(
        "Calibration contains missing signal observations."
    )

if calibration_data["record"].tolist() != list(
    range(CALIBRATION_START, MONITORING_START)
):

    raise ValueError(
        "Calibration records are not chronologically continuous."
    )

print("\nCalibration validation passed.")


# ============================================================
# 15. SAVE SIGNAL DIAGNOSTICS
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved signal diagnostics to:", OUTPUT_PATH)


# ============================================================
# 16. MONITORING SUMMARY
# ============================================================

monitoring = output[
    output["monitoring_phase"] == "monitoring"
].copy()

monitoring_signals = [
    "rolling_error",
    "error_increase",
    "rolling_confidence",
    "confidence_drop",
    "relationship_change"
]

monitoring = monitoring.dropna(
    subset=monitoring_signals
)

print("\n========== SIGNAL DIAGNOSTICS ==========")

print("Total evaluation records:", len(evaluation))

print("Warm-up records:", (
    output["monitoring_phase"] == "warmup"
).sum())

print("Calibration records:", (
    output["monitoring_phase"] == "calibration"
).sum())

print("Monitoring records:", len(monitoring))

print("Rolling window:", WINDOW)

print("\nMean signal values during monitoring:")

print(
    monitoring[monitoring_signals]
    .mean()
    .round(4)
)

print("\nFirst 5 monitoring records:")

print(
    monitoring.head()
    .to_string(index=False)
)

print("\nLast 5 monitoring records:")

print(
    monitoring.tail()
    .to_string(index=False)
)

print("\nElec2 signal analysis completed successfully.")
