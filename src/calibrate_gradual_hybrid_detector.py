
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# 1. PATHS AND EXPERIMENT SETTINGS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "gradual_hybrid_signal_diagnostics_v2.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"

VALIDATION_SEEDS = [42, 43, 44, 45, 46]

DRIFT_ONSET = 800
DRIFT_MIDPOINT = 1000
DRIFT_END = 1200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. LOAD AND PREPARE DATA
# ==========================================================

data = pd.read_csv(INPUT_FILE)

data = data.sort_values(
    ["seed", "record"]
).copy()

required_columns = [
    "seed",
    "record",
    "rolling_error",
    "rolling_confidence",
    "relationship_change",
    "reference_error",
]

missing = [
    col for col in required_columns
    if col not in data.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

available_seeds = sorted(data["seed"].unique())

if not set(VALIDATION_SEEDS).issubset(available_seeds):
    raise ValueError("One or more validation seeds are missing.")

# ==========================================================
# 3. CALCULATE THREE NORMALIZED SIGNALS
# ==========================================================

# Error signal
error_change = (
    data["rolling_error"] - data["reference_error"]
).clip(lower=0)

data["error_signal"] = (
    error_change / 0.50
).clip(0, 1)

# Relationship signal
data["relationship_signal"] = (
    data["relationship_change"] / 1.20
).clip(0, 1)

# Confidence signal: drop compared with 100 records earlier
data["previous_confidence"] = (
    data.groupby("seed")["rolling_confidence"]
    .shift(100)
)

confidence_drop = (
    data["previous_confidence"]
    - data["rolling_confidence"]
).clip(lower=0)

data["confidence_signal"] = (
    confidence_drop / 0.05
).clip(0, 1)

signal_columns = [
    "error_signal",
    "relationship_signal",
    "confidence_signal",
]

# Same valid observation window for both variants
data = data.dropna(subset=signal_columns).copy()

validation_data = data[
    data["seed"].isin(VALIDATION_SEEDS)
].copy()

print("Gradual-drift calibration data loaded.")
print("Validation seeds:", VALIDATION_SEEDS)
print("Valid records:", len(validation_data))

# ==========================================================
# 4. EVALUATE ONE CONFIGURATION
# ==========================================================

def evaluate_configuration(weights, threshold, variant):

    results = []

    for seed in VALIDATION_SEEDS:

        stream = validation_data[
            validation_data["seed"] == seed
        ].sort_values("record").copy()

        if variant == "two_signal":

            score = (
                weights[0] * stream["error_signal"]
                + weights[1] * stream["relationship_signal"]
            )

        else:

            score = (
                weights[0] * stream["error_signal"]
                + weights[1] * stream["confidence_signal"]
                + weights[2] * stream["relationship_signal"]
            )

        # Detect only rising threshold crossings.
        # Consecutive triggered records count as one episode.
        active = score >= threshold

        previous_active = active.shift(
            1, fill_value=False
        )

        alarm_episode = active & ~previous_active

        alarm_records = stream.loc[
            alarm_episode, "record"
        ].astype(int).tolist()

        # False alarms are episodes before drift onset.
        false_alarm_records = [
            record for record in alarm_records
            if record < DRIFT_ONSET
        ]

        # First alarm episode at or after drift onset.
        post_drift_alarms = [
            record for record in alarm_records
            if record >= DRIFT_ONSET
        ]

        if post_drift_alarms:

            first_detection = post_drift_alarms[0]

            detected = 1

            delay = first_detection - DRIFT_ONSET

        else:

            first_detection = None
            detected = 0
            delay = np.nan

        results.append({

            "seed": seed,

            "false_alarm_count": len(false_alarm_records),

            "false_alarm_records": false_alarm_records,

            "first_detection": first_detection,

            "detected": detected,

            "delay_from_onset": delay,

        })

    results_df = pd.DataFrame(results)

    return {

        "false_alarms": int(
            results_df["false_alarm_count"].sum()
        ),

        "detections": int(
            results_df["detected"].sum()
        ),

        "mean_delay": (
            float(results_df["delay_from_onset"].mean())
            if results_df["delay_from_onset"].notna().any()
            else np.inf
        ),

        "details": results_df,

    }


# ==========================================================
# 5. CALIBRATE TWO-SIGNAL DETECTOR
# ==========================================================

two_signal_results = []

thresholds = np.round(
    np.arange(0.05, 0.801, 0.025), 3
)

for error_weight_step in range(1, 10):

    error_weight = error_weight_step / 10

    relationship_weight = 1 - error_weight

    weights = [
        error_weight,
        relationship_weight,
    ]

    for threshold in thresholds:

        result = evaluate_configuration(
            weights,
            threshold,
            "two_signal",
        )

        two_signal_results.append({

            "variant": "two_signal",

            "error_weight": weights[0],

            "confidence_weight": 0.0,

            "relationship_weight": weights[1],

            "threshold": threshold,

            "false_alarms": result["false_alarms"],

            "detections": result["detections"],

            "mean_delay": result["mean_delay"],

        })


# ==========================================================
# 6. CALIBRATE THREE-SIGNAL DETECTOR
# ==========================================================

three_signal_results = []

# Each signal receives at least 0.10 weight.
# All weights sum to 1.

for error_step in range(1, 9):

    for confidence_step in range(1, 10 - error_step):

        relationship_step = (
            10 - error_step - confidence_step
        )

        if relationship_step < 1:
            continue

        weights = [
            error_step / 10,
            confidence_step / 10,
            relationship_step / 10,
        ]

        for threshold in thresholds:

            result = evaluate_configuration(
                weights,
                threshold,
                "three_signal",
            )

            three_signal_results.append({

                "variant": "three_signal",

                "error_weight": weights[0],

                "confidence_weight": weights[1],

                "relationship_weight": weights[2],

                "threshold": threshold,

                "false_alarms": result["false_alarms"],

                "detections": result["detections"],

                "mean_delay": result["mean_delay"],

            })


# ==========================================================
# 7. SAVE CALIBRATION GRID
# ==========================================================

all_results = pd.DataFrame(
    two_signal_results + three_signal_results
)

all_results["mean_delay"] = (
    all_results["mean_delay"].replace(np.inf, np.nan)
)

grid_path = (
    OUTPUT_DIR / "gradual_hybrid_calibration_grid.csv"
)

all_results.to_csv(grid_path, index=False)


# ==========================================================
# 8. SELECT BEST CONFIGURATIONS
# ==========================================================

def select_best_configuration(results, variant):

    candidates = results[
        results["variant"] == variant
    ].copy()

    # Priority 1:
    # Zero false alarms and detection on all validation seeds.

    preferred = candidates[
        (candidates["false_alarms"] == 0)
        &
        (candidates["detections"] == len(VALIDATION_SEEDS))
    ]

    if not preferred.empty:

        selected = preferred.sort_values(
            ["mean_delay", "threshold"],
            ascending=[True, False],
        ).iloc[0]

        selection_rule = (
            "Zero pre-drift false alarms and detection "
            "on all validation seeds; minimum mean delay."
        )

    else:

        # Fallback:
        # Minimize false alarms, maximize detections,
        # then minimize detection delay.

        candidates["mean_delay"] = (
            candidates["mean_delay"].fillna(np.inf)
        )

        selected = candidates.sort_values(
            [
                "false_alarms",
                "detections",
                "mean_delay",
                "threshold",
            ],
            ascending=[True, False, True, False],
        ).iloc[0]

        selection_rule = (
            "Fallback: minimize false alarms, maximize "
            "detections, then minimize mean delay."
        )

    return selected.to_dict(), selection_rule


selected_two, rule_two = select_best_configuration(
    all_results,
    "two_signal",
)

selected_three, rule_three = select_best_configuration(
    all_results,
    "three_signal",
)


# ==========================================================
# 9. SAVE SELECTED CONFIGURATIONS
# ==========================================================

selected_configurations = {

    "experiment": "gradual_drift",

    "validation_seeds": VALIDATION_SEEDS,

    "drift_onset": DRIFT_ONSET,

    "drift_midpoint": DRIFT_MIDPOINT,

    "drift_end": DRIFT_END,

    "signal_normalization": {

        "error_scale": 0.50,

        "relationship_scale": 1.20,

        "confidence_drop_scale": 0.05,

        "confidence_lag": 100,

    },

    "alarm_definition": (
        "Rising threshold crossing; consecutive triggered "
        "records count as one alarm episode."
    ),

    "two_signal": {

        "configuration": selected_two,

        "selection_rule": rule_two,

    },

    "three_signal": {

        "configuration": selected_three,

        "selection_rule": rule_three,

    },

}

config_path = (
    OUTPUT_DIR / "selected_gradual_hybrid_configurations.json"
)


def convert_numpy(value):

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    raise TypeError(f"Unsupported type: {type(value)}")


with open(config_path, "w") as file:

    json.dump(
        selected_configurations,
        file,
        indent=4,
        default=convert_numpy,
    )


# ==========================================================
# 10. DISPLAY RESULTS
# ==========================================================

print("\n========== GRADUAL HYBRID CALIBRATION COMPLETE ==========")

print("\nValidation seeds:", VALIDATION_SEEDS)

print("\nDrift onset:", DRIFT_ONSET)
print("Drift midpoint:", DRIFT_MIDPOINT)
print("Drift completion:", DRIFT_END)

print("\n--- SELECTED TWO-SIGNAL CONFIGURATION ---")

for key, value in selected_two.items():
    print(f"{key}: {value}")

print("\nSelection rule:", rule_two)

print("\n--- SELECTED THREE-SIGNAL CONFIGURATION ---")

for key, value in selected_three.items():
    print(f"{key}: {value}")

print("\nSelection rule:", rule_three)

print("\nSaved calibration grid:")
print(grid_path)

print("\nSaved selected configurations:")
print(config_path)
