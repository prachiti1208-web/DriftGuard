
import pandas as pd
import numpy as np

from pathlib import Path

from river.drift.binary import DDM, EDDM
from river.drift import ADWIN, PageHinkley


# -----------------------------------
# 1. Project configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_DIR = ROOT / "results" / "tables" / "streams"
OUTPUT_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = range(42, 52)

TRUE_DRIFT_POINT = 1000


# -----------------------------------
# 2. Detector configuration
# -----------------------------------

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


# -----------------------------------
# 3. Experiment results
# -----------------------------------

summary_results = []
alarm_results = []


# -----------------------------------
# 4. Run detectors on every seed
# -----------------------------------

for seed in SEEDS:

    print("\n" + "=" * 55)
    print(f"RUNNING DETECTORS FOR SEED {seed}")
    print("=" * 55)

    prediction_file = (
        PREDICTION_DIR /
        f"baseline_seed_{seed}.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file missing: {prediction_file}"
        )

    data = pd.read_csv(prediction_file)

    detectors = create_detectors()

    seed_results = {
        name: []
        for name in detectors
    }

    # Process the prediction error stream
    for _, row in data.iterrows():

        record = int(row["record"])

        error = int(
            row["actual"] != row["predicted"]
        )

        # Feed the same error to every detector
        for name, detector in detectors.items():

            detector.update(error)

            if detector.drift_detected:

                seed_results[name].append(record)

                alarm_results.append({
                    "seed": seed,
                    "detector": name,
                    "alarm_record": record,
                    "error": error
                })

    # -----------------------------------
    # 5. Evaluate each detector
    # -----------------------------------

    for name, alarms in seed_results.items():

        false_alarms = [
            point for point in alarms
            if point < TRUE_DRIFT_POINT
        ]

        valid_detections = [
            point for point in alarms
            if point >= TRUE_DRIFT_POINT
        ]

        first_detection = (
            min(valid_detections)
            if valid_detections
            else np.nan
        )

        detection_delay = (
            first_detection - TRUE_DRIFT_POINT
            if not np.isnan(first_detection)
            else np.nan
        )

        summary_results.append({

            "seed": seed,
            "detector": name,
            "true_drift_point": TRUE_DRIFT_POINT,
            "total_alarms": len(alarms),
            "false_alarm_count": len(false_alarms),
            "first_detection": first_detection,
            "detection_delay": detection_delay,
            "detected": len(valid_detections) > 0

        })

        print(
            f"{name}: "
            f"False alarms={len(false_alarms)}, "
            f"First detection={first_detection}, "
            f"Delay={detection_delay}"
        )


# -----------------------------------
# 6. Save results
# -----------------------------------

summary_df = pd.DataFrame(summary_results)
alarms_df = pd.DataFrame(alarm_results)

summary_path = (
    OUTPUT_DIR / "multi_seed_detector_summary.csv"
)

alarms_path = (
    OUTPUT_DIR / "multi_seed_detector_alarms.csv"
)

summary_df.to_csv(summary_path, index=False)
alarms_df.to_csv(alarms_path, index=False)


# -----------------------------------
# 7. Final summary
# -----------------------------------

print("\n" + "=" * 55)
print("MULTI-SEED DETECTOR EXPERIMENT COMPLETED")
print("=" * 55)

print("\nAverage detection delay by detector:")

print(
    summary_df.groupby("detector")["detection_delay"]
    .mean()
    .round(2)
)

print("\nAverage false alarms by detector:")

print(
    summary_df.groupby("detector")["false_alarm_count"]
    .mean()
    .round(2)
)

print("\nSaved files:")
print(summary_path)
print(alarms_path)
