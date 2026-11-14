
import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "multi_seed_hybrid_signal_diagnostics_v2.csv"
)

CONFIG_FILE = (
    ROOT / "results" / "tables"
    / "selected_hybrid_configurations_extended.json"
)

OUTPUT_DIR = ROOT / "results" / "tables"

TEST_SEEDS = [47, 48, 49, 50, 51]
TRUE_DRIFT_POINT = 1000


# --------------------------------------------------
# 2. LOAD FROZEN CONFIGURATIONS
# --------------------------------------------------

with open(CONFIG_FILE, "r") as file:
    saved_config = json.load(file)

normalization = saved_config["signal_normalization"]

ERROR_SCALE = normalization["error_scale"]
RELATIONSHIP_SCALE = normalization["relationship_scale"]
CONFIDENCE_DROP_SCALE = normalization["confidence_drop_scale"]
CONFIDENCE_LAG = normalization["confidence_lag"]

configurations = {
    "two_signal": saved_config["two_signal"]["configuration"],
    "three_signal": saved_config["three_signal"]["configuration"],
}


# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "seed",
    "record",
    "rolling_error",
    "reference_error",
    "rolling_confidence",
    "relationship_change",
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

test_data = data[data["seed"].isin(TEST_SEEDS)].copy()

if set(test_data["seed"].unique()) != set(TEST_SEEDS):
    raise ValueError("One or more test seeds are missing from the input file.")


# --------------------------------------------------
# 4. CALCULATE SIGNALS
# --------------------------------------------------

test_data = test_data.sort_values(["seed", "record"])

# Error signal
test_data["error_signal"] = (
    (test_data["rolling_error"] - test_data["reference_error"])
    .clip(lower=0)
    / ERROR_SCALE
).clip(0, 1)

# Relationship signal
test_data["relationship_signal"] = (
    test_data["relationship_change"] / RELATIONSHIP_SCALE
).clip(0, 1)

# Confidence signal: compare current confidence with
# confidence 100 records earlier within the same stream.
previous_confidence = (
    test_data.groupby("seed")["rolling_confidence"]
    .shift(CONFIDENCE_LAG)
)

test_data["confidence_signal"] = (
    (previous_confidence - test_data["rolling_confidence"])
    .clip(lower=0)
    / CONFIDENCE_DROP_SCALE
).clip(0, 1)

# Use the same valid evaluation period for both variants.
test_data = test_data.dropna(
    subset=[
        "error_signal",
        "relationship_signal",
        "confidence_signal",
    ]
).copy()


# --------------------------------------------------
# 5. EVALUATE EACH CONFIGURATION
# --------------------------------------------------

per_seed_results = []

for variant, config in configurations.items():

    print(f"\nEvaluating {variant}...")

    error_weight = config["error_weight"]
    confidence_weight = config["confidence_weight"]
    relationship_weight = config["relationship_weight"]
    threshold = config["threshold"]

    for seed in TEST_SEEDS:

        stream = (
            test_data[test_data["seed"] == seed]
            .sort_values("record")
            .copy()
        )

        if stream.empty:
            raise ValueError(f"No valid records for seed {seed}.")

        # Combined hybrid score
        stream["hybrid_score"] = (
            error_weight * stream["error_signal"]
            + confidence_weight * stream["confidence_signal"]
            + relationship_weight * stream["relationship_signal"]
        )

        # An alarm episode begins when the score crosses
        # from below threshold to at/above threshold.
        above_threshold = stream["hybrid_score"] >= threshold

        previous_above = above_threshold.shift(
            1, fill_value=False
        )

        stream["alarm_episode"] = (
            above_threshold & ~previous_above
        )

        alarms = stream[stream["alarm_episode"]].copy()

        # Count false alarm episodes before true drift.
        pre_drift_alarms = alarms[
            alarms["record"] < TRUE_DRIFT_POINT
        ]

        false_alarm_count = len(pre_drift_alarms)

        # Find the first NEW alarm episode after true drift.
        post_drift_alarms = alarms[
            alarms["record"] >= TRUE_DRIFT_POINT
        ]

        if not post_drift_alarms.empty:

            detection_record = int(
                post_drift_alarms.iloc[0]["record"]
            )

            detection_delay = (
                detection_record - TRUE_DRIFT_POINT
            )

            detected = 1

        else:

            detection_record = None
            detection_delay = None
            detected = 0

        valid_pre_drift_records = len(
            stream[stream["record"] < TRUE_DRIFT_POINT]
        )

        per_seed_results.append({
            "variant": variant,
            "seed": seed,
            "error_weight": error_weight,
            "confidence_weight": confidence_weight,
            "relationship_weight": relationship_weight,
            "threshold": threshold,
            "false_alarms": false_alarm_count,
            "detected": detected,
            "detection_record": detection_record,
            "detection_delay": detection_delay,
            "missed_detection": 1 - detected,
            "valid_pre_drift_records": valid_pre_drift_records,
        })

        print(
            f"Seed {seed}: "
            f"False alarms={false_alarm_count}, "
            f"Detected={detected}, "
            f"Delay={detection_delay}"
        )


# --------------------------------------------------
# 6. CREATE PER-SEED RESULTS
# --------------------------------------------------

per_seed_df = pd.DataFrame(per_seed_results)


# --------------------------------------------------
# 7. CALCULATE AGGREGATE METRICS
# --------------------------------------------------

summary_results = []

for variant in configurations:

    subset = per_seed_df[
        per_seed_df["variant"] == variant
    ]

    total_tp = int(subset["detected"].sum())
    total_fn = int(subset["missed_detection"].sum())
    total_fp = int(subset["false_alarms"].sum())

    total_pre_drift_records = int(
        subset["valid_pre_drift_records"].sum()
    )

    mean_delay = subset["detection_delay"].mean()
    median_delay = subset["detection_delay"].median()

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    false_alarm_rate = (
        total_fp / total_pre_drift_records * 1000
        if total_pre_drift_records > 0
        else 0
    )

    summary_results.append({
        "variant": variant,
        "test_streams": len(subset),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "missed_detections": total_fn,
        "mean_detection_delay": mean_delay,
        "median_detection_delay": median_delay,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarms_per_1000_records": false_alarm_rate,
    })


summary_df = pd.DataFrame(summary_results)


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

per_seed_path = OUTPUT_DIR / "hybrid_test_per_seed.csv"
summary_path = OUTPUT_DIR / "hybrid_test_summary.csv"

per_seed_df.to_csv(per_seed_path, index=False)
summary_df.to_csv(summary_path, index=False)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== HELD-OUT TEST: PER-SEED RESULTS ==========")
print(per_seed_df.to_string(index=False))

print("\n========== HELD-OUT TEST: SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\nResults saved:")
print(per_seed_path)
print(summary_path)
