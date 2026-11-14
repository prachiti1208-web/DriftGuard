
import pandas as pd
import numpy as np

from pathlib import Path

from river.drift.binary import DDM, EDDM
from river.drift import ADWIN, PageHinkley


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIR = (
    ROOT / "results" / "tables" / "speed_tests"
)

OUTPUT_DIR = (
    ROOT / "results" / "tables" / "speed_tests"
)

# New independent test seeds
TEST_SEEDS = list(range(52, 72))

DRIFT_ONSET = 800

DRIFT_TIMINGS = {
    "fast": {
        "midpoint": 900,
        "end": 1000
    },
    "medium": {
        "midpoint": 1000,
        "end": 1200
    },
    "slow": {
        "midpoint": 1200,
        "end": 1600
    }
}

SPEEDS = ["fast", "medium", "slow"]


# --------------------------------------------------
# 2. Create conventional detectors
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
# 3. Validate input files
# --------------------------------------------------

def validate_input_files():

    missing_files = []

    for speed in SPEEDS:

        for seed in TEST_SEEDS:

            input_file = (
                PREDICTION_DIR
                / f"{speed}_seed_{seed}_predictions.csv"
            )

            if not input_file.exists():
                missing_files.append(input_file)

    if missing_files:

        print("\nMissing prediction files:")

        for file in missing_files:
            print(file)

        raise FileNotFoundError(
            f"{len(missing_files)} prediction files are missing."
        )

    print(
        f"\nInput validation successful: "
        f"{len(SPEEDS) * len(TEST_SEEDS)} prediction files found."
    )


# --------------------------------------------------
# 4. Run detectors on one prediction stream
# --------------------------------------------------

def run_detectors(speed, seed):

    input_file = (
        PREDICTION_DIR
        / f"{speed}_seed_{seed}_predictions.csv"
    )

    predictions = pd.read_csv(input_file)

    required_columns = {
        "record",
        "actual",
        "predicted"
    }

    missing_columns = (
        required_columns - set(predictions.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns in {input_file}: "
            f"{missing_columns}"
        )

    predictions = predictions.sort_values(
        "record"
    )

    detectors = create_detectors()

    raw_alarms = []

    for _, row in predictions.iterrows():

        record = int(row["record"])

        error = int(
            row["actual"] != row["predicted"]
        )

        for detector_name, detector in detectors.items():

            detector.update(error)

            if detector.drift_detected:

                raw_alarms.append({
                    "speed": speed,
                    "seed": seed,
                    "detector": detector_name,
                    "alarm_record": record,
                    "error": error
                })

    return predictions, raw_alarms


# --------------------------------------------------
# 5. Group consecutive alarms into episodes
# --------------------------------------------------

def group_alarm_episodes(raw_alarms):

    columns = [
        "speed",
        "seed",
        "detector",
        "alarm_record",
        "error"
    ]

    if not raw_alarms:

        return pd.DataFrame(columns=columns)

    alarms = pd.DataFrame(raw_alarms)

    alarms = alarms.sort_values(
        ["speed", "seed", "detector", "alarm_record"]
    )

    episode_rows = []

    for (speed, seed, detector), group in alarms.groupby(
        ["speed", "seed", "detector"]
    ):

        previous_record = None

        for _, alarm in group.iterrows():

            current_record = int(
                alarm["alarm_record"]
            )

            # Consecutive alarm records belong
            # to the same alarm episode.
            if (
                previous_record is None
                or current_record > previous_record + 1
            ):

                episode_rows.append({
                    "speed": speed,
                    "seed": seed,
                    "detector": detector,
                    "alarm_record": current_record,
                    "error": alarm["error"]
                })

            previous_record = current_record

    return pd.DataFrame(episode_rows)


# --------------------------------------------------
# 6. Evaluate one detector on one stream
# --------------------------------------------------

def evaluate_detector(
    alarms,
    predictions,
    speed,
    seed,
    detector_name
):

    timing = DRIFT_TIMINGS[speed]

    midpoint = timing["midpoint"]
    drift_end = timing["end"]

    detector_alarms = alarms[
        (alarms["speed"] == speed)
        & (alarms["seed"] == seed)
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

        delay_onset = (
            detection_record - DRIFT_ONSET
        )

        delay_midpoint = (
            detection_record - midpoint
        )

        delay_completion = (
            detection_record - drift_end
        )

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
        "speed": speed,
        "seed": seed,
        "detector": detector_name,
        "detected": detected,
        "detection_record": detection_record,
        "false_alarm_episodes": false_alarm_count,
        "pre_drift_false_alarm": int(
            false_alarm_count > 0
        ),
        "delay_from_onset": delay_onset,
        "delay_from_midpoint": delay_midpoint,
        "delay_from_completion": delay_completion,
        "monitored_pre_drift_records": monitored_pre_drift
    }


# --------------------------------------------------
# 7. Aggregate performance
# --------------------------------------------------

def calculate_summary(results):

    summary_rows = []

    for (speed, detector_name), group in results.groupby(
        ["speed", "detector"]
    ):

        total_seeds = len(group)

        tp = int(group["detected"].sum())

        fn = total_seeds - tp

        fp = int(
            group["false_alarm_episodes"].sum()
        )

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

            "speed": speed,
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
# 8. Main experiment
# --------------------------------------------------

def main():

    print(
        "\n========== NEW SPEED-WISE CONVENTIONAL "
        "DETECTOR EXPERIMENT =========="
    )

    print("Test seeds:", TEST_SEEDS)
    print("Total seeds:", len(TEST_SEEDS))
    print("Drift onset:", DRIFT_ONSET)
    print("Drift timings:", DRIFT_TIMINGS)

    # Check that all 60 files exist before starting.
    validate_input_files()

    all_raw_alarms = []
    all_seed_results = []

    detector_names = [
        "DDM",
        "EDDM",
        "ADWIN",
        "Page-Hinkley"
    ]

    # Run detectors on all 60 streams.
    for speed in SPEEDS:

        print(
            f"\n========== {speed.upper()} DRIFT =========="
        )

        for seed in TEST_SEEDS:

            print(f"\nProcessing seed {seed}...")

            predictions, raw_alarms = run_detectors(
                speed,
                seed
            )

            all_raw_alarms.extend(raw_alarms)

            print("Raw alarms:", len(raw_alarms))

    # Convert raw alarms into episodes.
    episode_alarms = group_alarm_episodes(
        all_raw_alarms
    )

    # Evaluate each detector on every stream.
    for speed in SPEEDS:

        for seed in TEST_SEEDS:

            input_file = (
                PREDICTION_DIR
                / f"{speed}_seed_{seed}_predictions.csv"
            )

            predictions = pd.read_csv(input_file)

            for detector_name in detector_names:

                result = evaluate_detector(
                    episode_alarms,
                    predictions,
                    speed,
                    seed,
                    detector_name
                )

                all_seed_results.append(result)

    detailed_results = pd.DataFrame(
        all_seed_results
    )

    summary = calculate_summary(
        detailed_results
    )

    # --------------------------------------------------
    # 9. Save results using NEW filenames
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    alarm_path = (
        OUTPUT_DIR
        / "speed_conventional_detector_alarms_new_seeds.csv"
    )

    detailed_path = (
        OUTPUT_DIR
        / "speed_conventional_seed_results_new_seeds.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "speed_conventional_summary_new_seeds.csv"
    )

    episode_alarms.to_csv(
        alarm_path,
        index=False
    )

    detailed_results.to_csv(
        detailed_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # 10. Display results
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

    print(
        "\nNew speed-wise conventional detector experiment completed."
    )


if __name__ == "__main__":
    main()
