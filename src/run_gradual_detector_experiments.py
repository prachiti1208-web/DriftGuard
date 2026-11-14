
import pandas as pd
import numpy as np

from pathlib import Path

from river.drift.binary import DDM, EDDM
from river.drift import ADWIN, PageHinkley


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIR = ROOT / "results" / "tables"
OUTPUT_DIR = ROOT / "results" / "tables"

TEST_SEEDS = [47, 48, 49, 50, 51]

DRIFT_ONSET = 800
DRIFT_MIDPOINT = 1000
DRIFT_END = 1200


# --------------------------------------------------
# 2. Create conventional detectors
# Same parameters as abrupt-drift experiments
# --------------------------------------------------

def create_detectors():

    return {
        "DDM": DDM(
            warm_start=30,
            warning_threshold=2.0,
            drift_threshold=3.0
        ),

        "EDDM": EDDM(),

        "ADWIN": ADWIN(
            delta=0.002
        ),

        "Page-Hinkley": PageHinkley()
    }


# --------------------------------------------------
# 3. Run detectors on one prediction stream
# --------------------------------------------------

def run_detectors(seed):

    input_file = (
        PREDICTION_DIR
        / f"gradual_seed_{seed}_predictions.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {input_file}"
        )

    predictions = pd.read_csv(input_file)

    required_columns = {"record", "actual", "predicted"}

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_file}: {missing_columns}"
        )

    predictions = predictions.sort_values("record")

    detectors = create_detectors()

    raw_alarms = []

    for _, row in predictions.iterrows():

        record = int(row["record"])

        error = int(row["actual"] != row["predicted"])

        for detector_name, detector in detectors.items():

            detector.update(error)

            if detector.drift_detected:

                raw_alarms.append({
                    "seed": seed,
                    "detector": detector_name,
                    "alarm_record": record,
                    "error": error
                })

    return predictions, raw_alarms


# --------------------------------------------------
# 4. Group consecutive alarm records into episodes
# --------------------------------------------------

def group_alarm_episodes(raw_alarms):

    if not raw_alarms:
        return pd.DataFrame(
            columns=[
                "seed",
                "detector",
                "alarm_record",
                "error"
            ]
        )

    alarms = pd.DataFrame(raw_alarms)

    alarms = alarms.sort_values(
        ["seed", "detector", "alarm_record"]
    )

    episode_rows = []

    for (seed, detector), group in alarms.groupby(
        ["seed", "detector"]
    ):

        previous_record = None

        for _, alarm in group.iterrows():

            current_record = int(alarm["alarm_record"])

            # Consecutive alarm records belong to one episode.
            if (
                previous_record is None
                or current_record > previous_record + 1
            ):

                episode_rows.append({
                    "seed": seed,
                    "detector": detector,
                    "alarm_record": current_record,
                    "error": alarm["error"]
                })

            previous_record = current_record

    return pd.DataFrame(episode_rows)


# --------------------------------------------------
# 5. Evaluate detector performance
# --------------------------------------------------

def evaluate_detector(alarms, predictions, seed, detector_name):

    detector_alarms = alarms[
        (alarms["seed"] == seed)
        & (alarms["detector"] == detector_name)
    ].copy()

    pre_drift = detector_alarms[
        detector_alarms["alarm_record"] < DRIFT_ONSET
    ]

    post_onset = detector_alarms[
        detector_alarms["alarm_record"] >= DRIFT_ONSET
    ]

    false_alarm_count = len(pre_drift)

    if len(post_onset) > 0:

        detection_record = int(
            post_onset.iloc[0]["alarm_record"]
        )

        detected = 1

        delay_onset = detection_record - DRIFT_ONSET
        delay_midpoint = detection_record - DRIFT_MIDPOINT
        delay_completion = detection_record - DRIFT_END

    else:

        detection_record = np.nan
        detected = 0

        delay_onset = np.nan
        delay_midpoint = np.nan
        delay_completion = np.nan

    monitored_pre_drift = int(
        (predictions["record"] < DRIFT_ONSET).sum()
    )

    return {
        "seed": seed,
        "detector": detector_name,
        "detected": detected,
        "detection_record": detection_record,
        "false_alarm_episodes": false_alarm_count,
        "pre_drift_false_alarm": int(false_alarm_count > 0),
        "delay_from_onset": delay_onset,
        "delay_from_midpoint": delay_midpoint,
        "delay_from_completion": delay_completion,
        "monitored_pre_drift_records": monitored_pre_drift
    }


# --------------------------------------------------
# 6. Aggregate performance metrics
# --------------------------------------------------

def calculate_summary(results):

    summary_rows = []

    for detector_name, group in results.groupby("detector"):

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

        total_exposure = int(
            group["monitored_pre_drift_records"].sum()
        )

        false_alarms_per_1000 = (
            fp / total_exposure * 1000
            if total_exposure > 0
            else 0
        )

        summary_rows.append({

            "detector": detector_name,

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

            "mean_delay_from_midpoint":
                group["delay_from_midpoint"].mean(),

            "mean_delay_from_completion":
                group["delay_from_completion"].mean(),

            "false_alarms_per_1000_records":
                false_alarms_per_1000,

            "streams_with_pre_drift_false_alarm":
                int(group["pre_drift_false_alarm"].sum())
        })

    return pd.DataFrame(summary_rows)


# --------------------------------------------------
# 7. Main experiment
# --------------------------------------------------

def main():

    print("\n========== GRADUAL CONVENTIONAL DETECTOR TEST ==========")

    print("Test seeds:", TEST_SEEDS)
    print("Drift onset:", DRIFT_ONSET)
    print("Drift midpoint:", DRIFT_MIDPOINT)
    print("Drift completion:", DRIFT_END)

    all_raw_alarms = []
    all_seed_results = []

    for seed in TEST_SEEDS:

        print(f"\nProcessing seed {seed}...")

        predictions, raw_alarms = run_detectors(seed)

        all_raw_alarms.extend(raw_alarms)

        print("Raw alarms:", len(raw_alarms))

    # Convert raw alarms into event-level episodes.
    episode_alarms = group_alarm_episodes(all_raw_alarms)

    # Evaluate each detector separately for each seed.
    detector_names = [
        "DDM",
        "EDDM",
        "ADWIN",
        "Page-Hinkley"
    ]

    for seed in TEST_SEEDS:

        input_file = (
            PREDICTION_DIR
            / f"gradual_seed_{seed}_predictions.csv"
        )

        predictions = pd.read_csv(input_file)

        for detector_name in detector_names:

            result = evaluate_detector(
                episode_alarms,
                predictions,
                seed,
                detector_name
            )

            all_seed_results.append(result)

    detailed_results = pd.DataFrame(all_seed_results)

    summary = calculate_summary(detailed_results)

    # --------------------------------------------------
    # 8. Save results
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alarm_path = (
        OUTPUT_DIR / "gradual_detector_test_alarms.csv"
    )

    detailed_path = (
        OUTPUT_DIR / "gradual_detector_test_seed_results.csv"
    )

    summary_path = (
        OUTPUT_DIR / "gradual_detector_test_summary.csv"
    )

    episode_alarms.to_csv(alarm_path, index=False)

    detailed_results.to_csv(
        detailed_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # 9. Display results
    # --------------------------------------------------

    print("\n========== PER-SEED RESULTS ==========")

    print(
        detailed_results.to_string(index=False)
    )

    print("\n========== AGGREGATE PERFORMANCE ==========")

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved alarm episodes:")
    print(alarm_path)

    print("\nSaved detailed results:")
    print(detailed_path)

    print("\nSaved summary:")
    print(summary_path)

    print("\nGradual conventional detector experiment completed.")


if __name__ == "__main__":
    main()
