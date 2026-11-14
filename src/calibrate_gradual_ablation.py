
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"

INPUT_FILE = (
    TABLE_DIR / "gradual_hybrid_signal_diagnostics_v2.csv"
)

VALIDATION_SEEDS = [42, 43, 44, 45, 46]

DRIFT_ONSET = 800

ERROR_SCALE = 0.50
RELATIONSHIP_SCALE = 1.20
CONFIDENCE_SCALE = 0.05
CONFIDENCE_LAG = 100

THRESHOLDS = np.round(
    np.arange(0.05, 0.801, 0.025), 3
)

WEIGHT_VALUES = np.round(
    np.arange(0.1, 1.0, 0.1), 1
)


# --------------------------------------------------
# 2. Define ablation configurations
# --------------------------------------------------

VARIANTS = {
    "full_three_signal": [
        "error",
        "confidence",
        "relationship"
    ],

    "error_relationship": [
        "error",
        "relationship"
    ],

    "error_confidence": [
        "error",
        "confidence"
    ],

    "confidence_relationship": [
        "confidence",
        "relationship"
    ],

    "error_only": [
        "error"
    ],

    "confidence_only": [
        "confidence"
    ],

    "relationship_only": [
        "relationship"
    ]
}


# --------------------------------------------------
# 3. Load and prepare signals
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

data = data.sort_values(
    ["seed", "record"]
).copy()

data["error_signal"] = (
    (
        data["rolling_error"]
        - data["reference_error"]
    ).clip(lower=0) / ERROR_SCALE
).clip(0, 1)

data["relationship_signal"] = (
    data["relationship_change"]
    / RELATIONSHIP_SCALE
).clip(0, 1)

previous_confidence = (
    data.groupby("seed")["rolling_confidence"]
    .shift(CONFIDENCE_LAG)
)

data["confidence_signal"] = (
    (
        previous_confidence
        - data["rolling_confidence"]
    ).clip(lower=0) / CONFIDENCE_SCALE
).clip(0, 1)

# Use identical records for every ablation configuration.
# This prevents different monitoring exposure from
# influencing comparisons.

signal_columns = [
    "error_signal",
    "confidence_signal",
    "relationship_signal"
]

data = data.dropna(
    subset=signal_columns
).copy()

data = data[
    data["seed"].isin(VALIDATION_SEEDS)
].copy()

print("Ablation calibration data loaded.")
print("Validation seeds:", VALIDATION_SEEDS)
print("Common valid records:", len(data))


# --------------------------------------------------
# 4. Generate independent weight combinations
# --------------------------------------------------

def generate_weight_combinations(signals):

    if len(signals) == 1:
        return [{signals[0]: 1.0}]

    combinations = []

    for values in itertools.product(
        WEIGHT_VALUES,
        repeat=len(signals)
    ):

        if np.isclose(sum(values), 1.0):

            combinations.append({
                signal: float(weight)
                for signal, weight in zip(signals, values)
            })

    return combinations


# --------------------------------------------------
# 5. Evaluate one configuration
# --------------------------------------------------

def evaluate_configuration(
    df,
    signals,
    weights,
    threshold
):

    false_alarms = 0
    detections = 0
    delays = []

    for seed in VALIDATION_SEEDS:

        seed_data = df[
            df["seed"] == seed
        ].copy()

        score = np.zeros(len(seed_data))

        for signal in signals:

            column = f"{signal}_signal"

            score += (
                weights[signal]
                * seed_data[column].to_numpy()
            )

        active = score >= threshold

        previous_active = np.concatenate(
            ([False], active[:-1])
        )

        alarm_episodes = seed_data.loc[
            active & ~previous_active,
            "record"
        ].to_numpy()

        pre_drift_alarms = alarm_episodes[
            alarm_episodes < DRIFT_ONSET
        ]

        post_onset_alarms = alarm_episodes[
            alarm_episodes >= DRIFT_ONSET
        ]

        false_alarms += len(pre_drift_alarms)

        if len(post_onset_alarms) > 0:

            first_detection = int(
                post_onset_alarms[0]
            )

            detections += 1

            delays.append(
                first_detection - DRIFT_ONSET
            )

    mean_delay = (
        float(np.mean(delays))
        if delays
        else float("inf")
    )

    return {
        "false_alarms": int(false_alarms),
        "detections": int(detections),
        "mean_delay": mean_delay
    }


# --------------------------------------------------
# 6. Run calibration grid for all variants
# --------------------------------------------------

all_grid_results = []
selected_configurations = {}

for variant, signals in VARIANTS.items():

    print(f"\nCalibrating: {variant}")

    weight_combinations = (
        generate_weight_combinations(signals)
    )

    variant_results = []

    for weights in weight_combinations:

        for threshold in THRESHOLDS:

            metrics = evaluate_configuration(
                data,
                signals,
                weights,
                float(threshold)
            )

            result = {
                "variant": variant,
                "signals": "+".join(signals),
                "weights": json.dumps(weights),
                "threshold": float(threshold),
                **metrics
            }

            variant_results.append(result)
            all_grid_results.append(result)

    grid = pd.DataFrame(variant_results)

    # Preferred: zero false alarms and all seeds detected.
    preferred = grid[
        (grid["false_alarms"] == 0)
        & (grid["detections"] == len(VALIDATION_SEEDS))
    ]

    if len(preferred) > 0:

        selected = preferred.sort_values(
            ["mean_delay", "threshold"],
            ascending=[True, False]
        ).iloc[0]

        selection_rule = (
            "Zero false alarms and detection on all "
            "validation seeds; minimum mean delay."
        )

    else:

        # Fallback if strict criteria cannot be achieved.
        selected = grid.sort_values(
            [
                "false_alarms",
                "detections",
                "mean_delay",
                "threshold"
            ],
            ascending=[True, False, True, False]
        ).iloc[0]

        selection_rule = (
            "Fallback: minimize false alarms, maximize "
            "detections, then minimize mean delay."
        )

    selected_configurations[variant] = {
        "signals": signals,
        "weights": json.loads(selected["weights"]),
        "threshold": float(selected["threshold"]),
        "validation_false_alarms": int(
            selected["false_alarms"]
        ),
        "validation_detections": int(
            selected["detections"]
        ),
        "validation_mean_delay": float(
            selected["mean_delay"]
        ),
        "selection_rule": selection_rule
    }

    print("Selected weights:", selected["weights"])
    print("Selected threshold:", selected["threshold"])
    print("False alarms:", selected["false_alarms"])
    print("Detections:", selected["detections"])
    print("Mean delay:", round(
        selected["mean_delay"], 2
    ))


# --------------------------------------------------
# 7. Save calibration results
# --------------------------------------------------

grid_df = pd.DataFrame(all_grid_results)

grid_path = (
    TABLE_DIR / "gradual_ablation_calibration_grid.csv"
)

config_path = (
    TABLE_DIR / "selected_gradual_ablation_configurations.json"
)

grid_df.to_csv(grid_path, index=False)

output_config = {
    "experiment": "gradual_drift_ablation",
    "validation_seeds": VALIDATION_SEEDS,
    "drift_onset": DRIFT_ONSET,
    "signal_normalization": {
        "error_scale": ERROR_SCALE,
        "relationship_scale": RELATIONSHIP_SCALE,
        "confidence_drop_scale": CONFIDENCE_SCALE,
        "confidence_lag": CONFIDENCE_LAG
    },
    "common_complete_case_records": len(data),
    "alarm_definition": (
        "Rising threshold crossing; consecutive "
        "triggered records count as one alarm episode."
    ),
    "variants": selected_configurations
}

with open(config_path, "w") as file:

    json.dump(
        output_config,
        file,
        indent=4,
        allow_nan=False
    )


# --------------------------------------------------
# 8. Display selected configurations
# --------------------------------------------------

print("\n========== ABLATION CALIBRATION COMPLETE ==========")

for variant, config in selected_configurations.items():

    print(f"\n--- {variant.upper()} ---")
    print("Signals:", config["signals"])
    print("Weights:", config["weights"])
    print("Threshold:", config["threshold"])
    print(
        "Validation false alarms:",
        config["validation_false_alarms"]
    )
    print(
        "Validation detections:",
        config["validation_detections"]
    )
    print(
        "Validation mean delay:",
        round(config["validation_mean_delay"], 2)
    )

print("\nSaved calibration grid:")
print(grid_path)

print("\nSaved selected configurations:")
print(config_path)
