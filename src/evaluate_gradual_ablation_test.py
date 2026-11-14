
import json
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"

INPUT_FILE = (
    TABLE_DIR / "gradual_hybrid_signal_diagnostics_v2.csv"
)

CONFIG_FILE = (
    TABLE_DIR / "selected_gradual_ablation_configurations.json"
)

TEST_SEEDS = [47, 48, 49, 50, 51]


# --------------------------------------------------
# 2. Load calibration configuration
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

with open(CONFIG_FILE, "r") as file:
    saved_config = json.load(file)

DRIFT_ONSET = saved_config["drift_onset"]

normalization = saved_config["signal_normalization"]

ERROR_SCALE = normalization["error_scale"]
RELATIONSHIP_SCALE = normalization["relationship_scale"]
CONFIDENCE_SCALE = normalization["confidence_drop_scale"]
CONFIDENCE_LAG = normalization["confidence_lag"]

variants = saved_config["variants"]


# --------------------------------------------------
# 3. Prepare all three signals
# --------------------------------------------------

data = data.sort_values(["seed", "record"]).copy()

data["error_signal"] = (
    (
        data["rolling_error"] - data["reference_error"]
    ).clip(lower=0) / ERROR_SCALE
).clip(0, 1)

data["relationship_signal"] = (
    data["relationship_change"] / RELATIONSHIP_SCALE
).clip(0, 1)

previous_confidence = (
    data.groupby("seed")["rolling_confidence"]
    .shift(CONFIDENCE_LAG)
)

data["confidence_signal"] = (
    (
        previous_confidence - data["rolling_confidence"]
    ).clip(lower=0) / CONFIDENCE_SCALE
).clip(0, 1)


# --------------------------------------------------
# 4. Apply common complete-case evaluation records
# --------------------------------------------------

signal_columns = [
    "error_signal",
    "confidence_signal",
    "relationship_signal"
]

data = data.dropna(subset=signal_columns).copy()

data = data[
    data["seed"].isin(TEST_SEEDS)
].copy()

print("\nCommon complete-case test records:", len(data))
print("Test seeds:", TEST_SEEDS)
print("Drift onset:", DRIFT_ONSET)


# --------------------------------------------------
# 5. Evaluate one configuration
# --------------------------------------------------

def evaluate_variant(df, variant_name, config):

    signals = config["signals"]
    weights = config["weights"]
    threshold = config["threshold"]

    seed_results = []

    for seed in TEST_SEEDS:

        seed_data = df[
            df["seed"] == seed
        ].copy()

        # Calculate weighted score
        score = np.zeros(len(seed_data))

        for signal in signals:

            score += (
                weights[signal]
                * seed_data[f"{signal}_signal"].to_numpy()
            )

        seed_data["score"] = score

        # Rising threshold crossings = alarm episodes
        active = seed_data["score"] >= threshold

        previous_active = active.shift(
            1, fill_value=False
        )

        seed_data["alarm_episode"] = (
            active & ~previous_active
        )

        alarms = seed_data[
            seed_data["alarm_episode"]
        ]

        pre_drift_alarms = alarms[
            alarms["record"] < DRIFT_ONSET
        ]

        post_onset_alarms = alarms[
            alarms["record"] >= DRIFT_ONSET
        ]

        false_alarm_count = len(pre_drift_alarms)

        if len(post_onset_alarms) > 0:

            detection_record = int(
                post_onset_alarms.iloc[0]["record"]
            )

            detected = 1
            delay = detection_record - DRIFT_ONSET

        else:

            detection_record = np.nan
            detected = 0
            delay = np.nan

        monitored_records = int(
            (seed_data["record"] < DRIFT_ONSET).sum()
        )

        seed_results.append({

            "variant": variant_name,
            "seed": seed,
            "detected": detected,
            "detection_record": detection_record,
            "false_alarm_episodes": false_alarm_count,
            "pre_drift_false_alarm": int(
                false_alarm_count > 0
            ),
            "delay_from_onset": delay,
            "monitored_pre_drift_records": monitored_records

        })

    return pd.DataFrame(seed_results)


# --------------------------------------------------
# 6. Run all seven configurations
# --------------------------------------------------

all_results = []

for variant_name, config in variants.items():

    print(f"\nEvaluating {variant_name}...")

    results = evaluate_variant(
        data,
        variant_name,
        config
    )

    all_results.append(results)


detailed = pd.concat(
    all_results,
    ignore_index=True
)


# --------------------------------------------------
# 7. Calculate aggregate metrics
# --------------------------------------------------

summary_rows = []

for variant_name, group in detailed.groupby("variant"):

    total_seeds = len(group)

    tp = int(group["detected"].sum())
    fn = total_seeds - tp

    fp = int(group["false_alarm_episodes"].sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    exposure = int(
        group["monitored_pre_drift_records"].sum()
    )

    false_alarm_rate = (
        fp / exposure * 1000
        if exposure > 0
        else 0
    )

    summary_rows.append({

        "variant": variant_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,

        "mean_delay_from_onset":
            group["delay_from_onset"].mean(),

        "median_delay_from_onset":
            group["delay_from_onset"].median(),

        "false_alarms_per_1000_records":
            false_alarm_rate,

        "streams_with_pre_drift_false_alarm":
            int(group["pre_drift_false_alarm"].sum())

    })


summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["mean_delay_from_onset", "F1"],
    ascending=[True, False]
).reset_index(drop=True)


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

detailed_path = (
    TABLE_DIR / "gradual_ablation_test_seed_results.csv"
)

summary_path = (
    TABLE_DIR / "gradual_ablation_test_summary.csv"
)

detailed.to_csv(detailed_path, index=False)
summary.to_csv(summary_path, index=False)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========== GRADUAL ABLATION HELD-OUT TEST ==========")

print("\n--- PER-SEED RESULTS ---")

print(
    detailed.round(4).to_string(index=False)
)

print("\n--- AGGREGATE ABLATION RESULTS ---")

print(
    summary.round(4).to_string(index=False)
)

print("\nSaved detailed results:")
print(detailed_path)

print("\nSaved summary:")
print(summary_path)

print("\nGradual ablation held-out evaluation completed.")
