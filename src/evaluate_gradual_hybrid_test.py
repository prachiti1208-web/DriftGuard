
import json
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "gradual_hybrid_signal_diagnostics_v2.csv"
)

CONFIG_FILE = (
    ROOT / "results" / "tables"
    / "selected_gradual_hybrid_configurations.json"
)

OUTPUT_DIR = ROOT / "results" / "tables"

TEST_SEEDS = [47, 48, 49, 50, 51]


# --------------------------------------------------
# 2. Load data and calibrated configurations
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

with open(CONFIG_FILE, "r") as file:
    saved_config = json.load(file)

onset = saved_config["drift_onset"]
midpoint = saved_config["drift_midpoint"]
drift_end = saved_config["drift_end"]

normalization = saved_config["signal_normalization"]

ERROR_SCALE = normalization["error_scale"]
RELATIONSHIP_SCALE = normalization["relationship_scale"]
CONFIDENCE_SCALE = normalization["confidence_drop_scale"]
CONFIDENCE_LAG = normalization["confidence_lag"]

two_config = saved_config["two_signal"]["configuration"]
three_config = saved_config["three_signal"]["configuration"]


# --------------------------------------------------
# 3. Prepare signal scores
# --------------------------------------------------

data = data.sort_values(["seed", "record"]).copy()

data["error_signal"] = (
    (data["rolling_error"] - data["reference_error"]).clip(lower=0)
    / ERROR_SCALE
).clip(0, 1)

data["relationship_signal"] = (
    data["relationship_change"] / RELATIONSHIP_SCALE
).clip(0, 1)

previous_confidence = (
    data.groupby("seed")["rolling_confidence"]
    .shift(CONFIDENCE_LAG)
)

data["confidence_signal"] = (
    (previous_confidence - data["rolling_confidence"]).clip(lower=0)
    / CONFIDENCE_SCALE
).clip(0, 1)


# --------------------------------------------------
# 4. Calculate hybrid scores
# --------------------------------------------------

data["two_signal_score"] = (
    two_config["error_weight"] * data["error_signal"]
    + two_config["relationship_weight"] * data["relationship_signal"]
)

data["three_signal_score"] = (
    three_config["error_weight"] * data["error_signal"]
    + three_config["confidence_weight"] * data["confidence_signal"]
    + three_config["relationship_weight"] * data["relationship_signal"]
)


# --------------------------------------------------
# 5. Evaluate one detector configuration
# --------------------------------------------------

def evaluate_configuration(df, config, score_column):

    threshold = config["threshold"]

    results = []

    for seed in TEST_SEEDS:

        seed_data = df[df["seed"] == seed].copy()
        seed_data = seed_data.dropna(subset=[score_column])

        active = seed_data[score_column] >= threshold

        previous_active = active.shift(1, fill_value=False)

        seed_data["alarm_episode"] = active & ~previous_active

        alarms = seed_data[seed_data["alarm_episode"]].copy()

        # False alarms before drift onset
        pre_drift_alarms = alarms[
            alarms["record"] < onset
        ]

        # First alarm at or after drift onset
        post_onset_alarms = alarms[
            alarms["record"] >= onset
        ]

        false_alarm_count = len(pre_drift_alarms)

        if len(post_onset_alarms) > 0:

            detection_record = int(
                post_onset_alarms.iloc[0]["record"]
            )

            detected = 1

            delay_onset = detection_record - onset
            delay_midpoint = detection_record - midpoint
            delay_completion = detection_record - drift_end

        else:

            detection_record = np.nan
            detected = 0

            delay_onset = np.nan
            delay_midpoint = np.nan
            delay_completion = np.nan

        monitored_pre_drift = int(
            (seed_data["record"] < onset).sum()
        )

        results.append({
            "seed": seed,
            "detected": detected,
            "detection_record": detection_record,
            "false_alarm_episodes": false_alarm_count,
            "pre_drift_false_alarm": int(false_alarm_count > 0),
            "delay_from_onset": delay_onset,
            "delay_from_midpoint": delay_midpoint,
            "delay_from_completion": delay_completion,
            "monitored_pre_drift_records": monitored_pre_drift
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. Evaluate both configurations
# --------------------------------------------------

two_results = evaluate_configuration(
    data, two_config, "two_signal_score"
)

three_results = evaluate_configuration(
    data, three_config, "three_signal_score"
)


# --------------------------------------------------
# 7. Calculate aggregate metrics
# --------------------------------------------------

def calculate_metrics(results, variant):

    total_seeds = len(results)

    tp = int(results["detected"].sum())
    fn = total_seeds - tp

    fp = int(results["false_alarm_episodes"].sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    total_exposure = int(
        results["monitored_pre_drift_records"].sum()
    )

    false_alarms_per_1000 = (
        fp / total_exposure * 1000
        if total_exposure > 0
        else 0
    )

    return {
        "variant": variant,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_delay_from_onset": results[
            "delay_from_onset"
        ].mean(),
        "median_delay_from_onset": results[
            "delay_from_onset"
        ].median(),
        "mean_delay_from_midpoint": results[
            "delay_from_midpoint"
        ].mean(),
        "mean_delay_from_completion": results[
            "delay_from_completion"
        ].mean(),
        "false_alarms_per_1000_records": false_alarms_per_1000,
        "streams_with_pre_drift_false_alarm": int(
            results["pre_drift_false_alarm"].sum()
        )
    }


summary = pd.DataFrame([
    calculate_metrics(two_results, "two_signal"),
    calculate_metrics(three_results, "three_signal")
])


# --------------------------------------------------
# 8. Save detailed and summary results
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

two_results.insert(0, "variant", "two_signal")
three_results.insert(0, "variant", "three_signal")

detailed_results = pd.concat(
    [two_results, three_results],
    ignore_index=True
)

detailed_path = (
    OUTPUT_DIR / "gradual_hybrid_test_seed_results.csv"
)

summary_path = (
    OUTPUT_DIR / "gradual_hybrid_test_summary.csv"
)

detailed_results.to_csv(detailed_path, index=False)
summary.to_csv(summary_path, index=False)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========== GRADUAL HYBRID HELD-OUT TEST ==========")

print("\nTest seeds:", TEST_SEEDS)
print("Drift onset:", onset)
print("Drift midpoint:", midpoint)
print("Drift completion:", drift_end)

print("\n--- PER-SEED RESULTS ---")
print(detailed_results.to_string(index=False))

print("\n--- AGGREGATE PERFORMANCE ---")
print(summary.round(4).to_string(index=False))

print("\nSaved detailed results:")
print(detailed_path)

print("\nSaved summary:")
print(summary_path)

print("\nGradual hybrid held-out evaluation completed.")
