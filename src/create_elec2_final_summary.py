
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

predictions_path = (
    ROOT / "results/tables/elec2_baseline_predictions.csv"
)

monitoring_path = (
    ROOT / "results/tables/elec2_driftguard_monitoring.csv"
)

early_warning_path = (
    ROOT / "results/tables/elec2_early_warning_validation.csv"
)

config_path = (
    ROOT / "results/tables/elec2_driftguard_config.json"
)

output_path = (
    ROOT / "results/tables/elec2_final_research_summary.csv"
)

# --------------------------------------------------
# 2. LOAD FILES
# --------------------------------------------------

predictions = pd.read_csv(predictions_path)
monitoring = pd.read_csv(monitoring_path)
early_warning = pd.read_csv(early_warning_path)

with open(config_path, "r") as f:
    config = json.load(f)

# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_prediction_columns = {"record", "correct"}
required_monitoring_columns = {
    "record", "alert", "hybrid_score"
}
required_early_columns = {
    "alert_at_start", "future_accuracy"
}

if not required_prediction_columns.issubset(predictions.columns):
    raise ValueError(
        f"Missing prediction columns: "
        f"{required_prediction_columns - set(predictions.columns)}"
    )

if not required_monitoring_columns.issubset(monitoring.columns):
    raise ValueError(
        f"Missing monitoring columns: "
        f"{required_monitoring_columns - set(monitoring.columns)}"
    )

if not required_early_columns.issubset(early_warning.columns):
    raise ValueError(
        f"Missing early-warning columns: "
        f"{required_early_columns - set(early_warning.columns)}"
    )

# --------------------------------------------------
# 4. BASELINE PERFORMANCE
# --------------------------------------------------

baseline_accuracy = (
    predictions["correct"].astype(bool).mean() * 100
)

# --------------------------------------------------
# 5. DRIFTGUARD MONITORING SUMMARY
# --------------------------------------------------

monitoring["alert"] = (
    monitoring["alert"].astype(bool)
)

total_records = len(monitoring)

alert_records = int(
    monitoring["alert"].sum()
)

alert_percentage = (
    alert_records / total_records * 100
)

# --------------------------------------------------
# 6. ALERT VS NON-ALERT ACCURACY
# --------------------------------------------------

comparison = monitoring.merge(
    predictions[["record", "correct"]].rename(
        columns={"correct": "prediction_correct"}
    ),
    on="record",
    how="left",
    validate="one_to_one"
)

if comparison["prediction_correct"].isna().any():
    raise ValueError(
        "Some monitoring records have no matching prediction."
    )

comparison["prediction_correct"] = (
    comparison["prediction_correct"].astype(bool)
)

alert_accuracy = (
    comparison.loc[
        comparison["alert"],
        "prediction_correct"
    ].mean() * 100
)

non_alert_accuracy = (
    comparison.loc[
        ~comparison["alert"],
        "prediction_correct"
    ].mean() * 100
)

# --------------------------------------------------
# 7. FUTURE PERFORMANCE VALIDATION
# --------------------------------------------------

alert_windows = early_warning[
    early_warning["alert_at_start"].astype(bool)
]

normal_windows = early_warning[
    ~early_warning["alert_at_start"].astype(bool)
]

if len(alert_windows) == 0 or len(normal_windows) == 0:
    raise ValueError(
        "Both alert and non-alert future windows are required."
    )

future_alert_accuracy = (
    alert_windows["future_accuracy"].mean()
)

future_normal_accuracy = (
    normal_windows["future_accuracy"].mean()
)

future_accuracy_difference = (
    future_normal_accuracy - future_alert_accuracy
)

# --------------------------------------------------
# 8. FINAL CONSOLIDATED SUMMARY
# --------------------------------------------------

summary = pd.DataFrame([
    [
        "Baseline prequential accuracy (%)",
        baseline_accuracy
    ],
    [
        "Total monitoring records",
        total_records
    ],
    [
        "Alert records",
        alert_records
    ],
    [
        "Alert percentage (%)",
        alert_percentage
    ],
    [
        "Accuracy during alerts (%)",
        alert_accuracy
    ],
    [
        "Accuracy without alerts (%)",
        non_alert_accuracy
    ],
    [
        "Future accuracy after alert (%)",
        future_alert_accuracy
    ],
    [
        "Future accuracy after no alert (%)",
        future_normal_accuracy
    ],
    [
        "Future accuracy difference (percentage points)",
        future_accuracy_difference
    ],
    [
        "Alert-start windows",
        len(alert_windows)
    ],
    [
        "Non-alert-start windows",
        len(normal_windows)
    ],
    [
        "Frozen hybrid threshold",
        config["threshold"]
    ]
], columns=["Metric", "Value"])

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(output_path, index=False)

print("\n========== FINAL ELEC2 RESEARCH SUMMARY ==========")
print(summary.to_string(index=False))

print("\nAll validation checks passed.")

print(f"\nSaved summary to:\n{output_path}")
