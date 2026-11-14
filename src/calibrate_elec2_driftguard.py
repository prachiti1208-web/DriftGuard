
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PATH = "results/tables/elec2_signal_diagnostics.csv"

OUTPUT_PATH = (
    "results/tables/elec2_driftguard_config.json"
)

SIGNALS = [
    "error_increase",
    "confidence_drop",
    "relationship_change"
]

# Keep the research framework's existing weights
WEIGHTS = {
    "error_increase": 0.60,
    "confidence_drop": 0.10,
    "relationship_change": 0.30
}

CALIBRATION_START = 1199
SCALING_END = 1298

THRESHOLD_START = 1299
THRESHOLD_END = 1398

THRESHOLD_QUANTILE = 0.95


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "record",
    "monitoring_phase"
] + SIGNALS

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 3. EXTRACT CALIBRATION SUBSETS
# ============================================================

scaling_data = df[
    df["record"].between(
        CALIBRATION_START,
        SCALING_END
    )
].copy()

threshold_data = df[
    df["record"].between(
        THRESHOLD_START,
        THRESHOLD_END
    )
].copy()


print("\n========== CALIBRATION DATA ==========")

print(
    "Scaling records:",
    scaling_data["record"].min(),
    "to",
    scaling_data["record"].max()
)

print("Scaling observations:", len(scaling_data))

print(
    "Threshold records:",
    threshold_data["record"].min(),
    "to",
    threshold_data["record"].max()
)

print("Threshold observations:", len(threshold_data))


# ============================================================
# 4. VALIDATE CALIBRATION DATA
# ============================================================

if len(scaling_data) != 100:
    raise ValueError(
        "Expected 100 scaling observations."
    )

if len(threshold_data) != 100:
    raise ValueError(
        "Expected 100 threshold observations."
    )

if scaling_data[SIGNALS].isna().any().any():
    raise ValueError(
        "Missing values in scaling data."
    )

if threshold_data[SIGNALS].isna().any().any():
    raise ValueError(
        "Missing values in threshold data."
    )

if not (
    scaling_data["monitoring_phase"] == "calibration"
).all():
    raise ValueError(
        "Scaling subset contains records outside calibration."
    )

if not (
    threshold_data["monitoring_phase"] == "calibration"
).all():
    raise ValueError(
        "Threshold subset contains records outside calibration."
    )


# ============================================================
# 5. ESTIMATE SIGNAL SCALING PARAMETERS
# ============================================================

signal_parameters = {}

for signal in SIGNALS:

    mean_value = float(
        scaling_data[signal].mean()
    )

    std_value = float(
        scaling_data[signal].std(ddof=0)
    )

    if not np.isfinite(std_value) or std_value < 1e-8:
        raise ValueError(
            f"Invalid standard deviation for {signal}"
        )

    signal_parameters[signal] = {
        "mean": mean_value,
        "std": std_value
    }


print("\n========== SIGNAL SCALING PARAMETERS ==========")

for signal, parameters in signal_parameters.items():

    print(f"\n{signal}")

    print("Mean:", round(parameters["mean"], 5))

    print("Std:", round(parameters["std"], 5))


# ============================================================
# 6. CALCULATE STANDARDIZED SIGNAL SCORES
# ============================================================

def calculate_scores(data):

    scores = pd.DataFrame(index=data.index)

    for signal in SIGNALS:

        mean_value = signal_parameters[signal]["mean"]

        std_value = signal_parameters[signal]["std"]

        z_score = (
            data[signal] - mean_value
        ) / std_value

        # Only increases in drift-related signals
        # contribute positively to the score.
        scores[signal] = z_score.clip(lower=0)

    return scores


# ============================================================
# 7. COMBINE THE THREE SIGNALS
# ============================================================

threshold_scores = calculate_scores(threshold_data)

threshold_scores["hybrid_score"] = sum(
    WEIGHTS[signal] * threshold_scores[signal]
    for signal in SIGNALS
)


# ============================================================
# 8. SELECT ALERT THRESHOLD
# ============================================================

alert_threshold = float(
    threshold_scores["hybrid_score"].quantile(
        THRESHOLD_QUANTILE
    )
)

if not np.isfinite(alert_threshold):
    raise ValueError(
        "Invalid hybrid alert threshold."
    )


print("\n========== THRESHOLD CALIBRATION ==========")

print("Weights:", WEIGHTS)

print(
    "Threshold quantile:",
    THRESHOLD_QUANTILE
)

print(
    "Selected alert threshold:",
    round(alert_threshold, 5)
)

print("\nThreshold subset score statistics:")

print(
    threshold_scores["hybrid_score"]
    .describe()
    .round(5)
)


# ============================================================
# 9. CALIBRATION ALERT DIAGNOSTICS
# ============================================================

threshold_alerts = (
    threshold_scores["hybrid_score"] > alert_threshold
)

print("\n========== CALIBRATION ALERT CHECK ==========")

print(
    "Threshold subset observations:",
    len(threshold_scores)
)

print(
    "Threshold subset records above threshold:",
    int(threshold_alerts.sum())
)

print(
    "Threshold subset record-level exceedance rate:",
    round(float(threshold_alerts.mean()), 4)
)


# ============================================================
# 10. SAVE FROZEN CONFIGURATION
# ============================================================

configuration = {
    "dataset": "Elec2",
    "method": "positive_zscore_weighted_hybrid",

    "signals": SIGNALS,

    "weights": WEIGHTS,

    "signal_parameters": signal_parameters,

    "threshold": alert_threshold,

    "threshold_quantile": THRESHOLD_QUANTILE,

    "scaling_records": [
        CALIBRATION_START,
        SCALING_END
    ],

    "threshold_records": [
        THRESHOLD_START,
        THRESHOLD_END
    ],

    "monitoring_start": 1399,

    "notes": (
        "Configuration calibrated using chronological "
        "Elec2 calibration observations. No monitoring "
        "records were used for parameter estimation."
    )
}

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w") as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


print("\n========== CONFIGURATION SAVED ==========")

print(OUTPUT_PATH)

print("\nElec2 DriftGuard calibration completed.")
