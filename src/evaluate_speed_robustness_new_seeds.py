
import json
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables" / "speed_tests"
    / "speed_test_hybrid_signal_diagnostics_new_seeds.csv"
)

CONFIG_FILE = (
    ROOT / "results" / "tables"
    / "selected_gradual_ablation_configurations.json"
)

OUTPUT_DIR = ROOT / "results" / "tables" / "speed_tests"

ONSET = 800

MIDPOINTS = {
    "fast": 900,
    "medium": 1000,
    "slow": 1200
}

COMPLETIONS = {
    "fast": 1000,
    "medium": 1200,
    "slow": 1600
}

TEST_SEEDS = range(52, 72)


# -----------------------------------------
# 2. LOAD FIXED CONFIGURATION
# -----------------------------------------

with open(CONFIG_FILE, "r") as file:
    config_data = json.load(file)

config = config_data["variants"]["full_three_signal"]

weights = config["weights"]
threshold = config["threshold"]

normalization = config_data["signal_normalization"]

ERROR_SCALE = normalization["error_scale"]
RELATIONSHIP_SCALE = normalization["relationship_scale"]
CONFIDENCE_SCALE = normalization["confidence_drop_scale"]
CONFIDENCE_LAG = normalization["confidence_lag"]


# -----------------------------------------
# 3. NORMALIZE SIGNALS
# -----------------------------------------

def calculate_hybrid_score(data):

    data = data.copy()

    data["error_signal"] = (
        (data["rolling_error"] - data["reference_error"])
        .clip(lower=0) / ERROR_SCALE
    ).clip(0, 1)

    data["relationship_signal"] = (
        data["relationship_change"]
        / RELATIONSHIP_SCALE
    ).clip(0, 1)

    data["confidence_signal"] = (
        (
            data["rolling_confidence"].shift(CONFIDENCE_LAG)
            - data["rolling_confidence"]
        ).clip(lower=0)
        / CONFIDENCE_SCALE
    ).clip(0, 1)

    data["hybrid_score"] = (
        weights["error"] * data["error_signal"]
        + weights["confidence"] * data["confidence_signal"]
        + weights["relationship"] * data["relationship_signal"]
    )

    return data


# -----------------------------------------
# 4. DETECT ALARM EPISODES
# -----------------------------------------

def get_alarm_episodes(data):

    triggered = (
        data["hybrid_score"] >= threshold
    ).fillna(False)

    previous = triggered.shift(1, fill_value=False)

    rising_crossings = triggered & ~previous

    return data.loc[rising_crossings, "record"].tolist()


# -----------------------------------------
# 5. EVALUATE ONE STREAM
# -----------------------------------------

def evaluate_stream(data, speed, seed):

    data = calculate_hybrid_score(data)

    # Common complete-case monitoring records
    data = data.dropna(
        subset=[
            "error_signal",
            "confidence_signal",
            "relationship_signal",
            "hybrid_score"
        ]
    ).reset_index(drop=True)

    alarms = get_alarm_episodes(data)

    pre_drift_alarms = [
        alarm for alarm in alarms if alarm < ONSET
    ]

    post_drift_alarms = [
        alarm for alarm in alarms if alarm >= ONSET
    ]

    detected = len(post_drift_alarms) > 0

    detection_record = (
        post_drift_alarms[0] if detected else np.nan
    )

    delay = (
        detection_record - ONSET if detected else np.nan
    )

    midpoint = MIDPOINTS[speed]
    completion = COMPLETIONS[speed]

    return {
        "speed": speed,
        "seed": seed,
        "TP": int(detected),
        "FP": len(pre_drift_alarms),
        "FN": int(not detected),
        "detection_record": detection_record,
        "delay_from_onset": delay,
        "delay_from_midpoint": (
            detection_record - midpoint if detected else np.nan
        ),
        "delay_from_completion": (
            detection_record - completion if detected else np.nan
        ),
        "pre_drift_false_alarms": len(pre_drift_alarms),
        "monitored_pre_drift_records": int(
            (data["record"] < ONSET).sum()
        ),
        "alarms": alarms
    }


# -----------------------------------------
# 6. MAIN EXPERIMENT
# -----------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)

    seed_results = []

    for speed in ["fast", "medium", "slow"]:

        print(f"\nEvaluating {speed.upper()} DRIFT")

        for seed in TEST_SEEDS:

            stream = data[
                (data["speed"] == speed)
                & (data["seed"] == seed)
            ].copy()

            if stream.empty:
                raise ValueError(
                    f"Missing stream: {speed}, seed {seed}"
                )

            result = evaluate_stream(stream, speed, seed)

            seed_results.append(result)

            print(
                f"Seed {seed}: "
                f"Detected={result['TP']}, "
                f"False alarms={result['FP']}, "
                f"Detection record={result['detection_record']}, "
                f"Delay={result['delay_from_onset']}"
            )

    results = pd.DataFrame(seed_results)

    # -------------------------------------
    # 7. SUMMARY METRICS
    # -------------------------------------

    summaries = []

    for speed in ["fast", "medium", "slow"]:

        group = results[results["speed"] == speed]

        TP = group["TP"].sum()
        FP = group["FP"].sum()
        FN = group["FN"].sum()

        precision = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        total_exposure = group[
            "monitored_pre_drift_records"
        ].sum()

        total_false_alarms = group[
            "pre_drift_false_alarms"
        ].sum()

        false_alarms_per_1000 = (
            total_false_alarms / total_exposure * 1000
            if total_exposure else 0
        )

        summaries.append({
            "speed": speed,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "mean_delay_from_onset": (
                group["delay_from_onset"].mean()
            ),
            "median_delay_from_onset": (
                group["delay_from_onset"].median()
            ),
            "mean_delay_from_midpoint": (
                group["delay_from_midpoint"].mean()
            ),
            "mean_delay_from_completion": (
                group["delay_from_completion"].mean()
            ),
            "false_alarms_per_1000_records": (
                false_alarms_per_1000
            ),
            "streams_with_pre_drift_false_alarm": (
                (group["pre_drift_false_alarms"] > 0).sum()
            )
        })

    summary = pd.DataFrame(summaries)

    # -------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------

    seed_output = (
        OUTPUT_DIR / "speed_robustness_new_seeds_seed_results.csv"
    )

    summary_output = (
        OUTPUT_DIR / "speed_robustness_new_seeds_summary.csv"
    )

    results.to_csv(seed_output, index=False)
    summary.to_csv(summary_output, index=False)

    print("\n========== NEW SEED SPEED ROBUSTNESS SUMMARY ==========")

    print(summary.to_string(index=False))

    print("\nFixed configuration:")
    print("Weights:", weights)
    print("Threshold:", threshold)

    print("\nFiles saved successfully:")
    print(seed_output)
    print(summary_output)


if __name__ == "__main__":
    main()
