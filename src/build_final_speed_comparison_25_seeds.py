
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results" / "tables" / "speed_tests"

# --------------------------------------------------
# 1. Input files
# --------------------------------------------------

FILES = {
    "hybrid_old": "speed_robustness_seed_results.csv",
    "hybrid_new": "speed_robustness_new_seeds_seed_results.csv",
    "conventional_old": "speed_conventional_seed_results.csv",
    "conventional_new": "speed_conventional_seed_results_new_seeds.csv",
}

# --------------------------------------------------
# 2. Load and standardize DriftGuard results
# --------------------------------------------------

hybrid_old = pd.read_csv(RESULTS_DIR / FILES["hybrid_old"])
hybrid_new = pd.read_csv(RESULTS_DIR / FILES["hybrid_new"])

hybrid = pd.concat(
    [hybrid_old, hybrid_new],
    ignore_index=True
)

hybrid["detector"] = "DriftGuard"

hybrid["detected"] = hybrid["TP"].astype(int)

hybrid["false_alarm_episodes"] = hybrid["FP"].astype(int)

hybrid["pre_drift_false_alarm"] = (
    hybrid["pre_drift_false_alarms"] > 0
).astype(int)

# --------------------------------------------------
# 3. Load conventional detector results
# --------------------------------------------------

conventional_old = pd.read_csv(
    RESULTS_DIR / FILES["conventional_old"]
)

conventional_new = pd.read_csv(
    RESULTS_DIR / FILES["conventional_new"]
)

conventional = pd.concat(
    [conventional_old, conventional_new],
    ignore_index=True
)

# --------------------------------------------------
# 4. Select common columns
# --------------------------------------------------

columns = [
    "speed",
    "seed",
    "detector",
    "detected",
    "detection_record",
    "false_alarm_episodes",
    "pre_drift_false_alarm",
    "delay_from_onset",
    "delay_from_midpoint",
    "delay_from_completion",
    "monitored_pre_drift_records"
]

hybrid = hybrid[columns]
conventional = conventional[columns]

# --------------------------------------------------
# 5. Combine all detector results
# --------------------------------------------------

combined = pd.concat(
    [hybrid, conventional],
    ignore_index=True
)

combined = combined.sort_values(
    ["speed", "detector", "seed"]
).reset_index(drop=True)

# --------------------------------------------------
# 6. Validate experiment coverage
# --------------------------------------------------

expected_seeds = set(range(47, 72))

print("\n========== DATA VALIDATION ==========")

for detector in combined["detector"].unique():

    detector_data = combined[
        combined["detector"] == detector
    ]

    print(f"\nDetector: {detector}")

    for speed in ["fast", "medium", "slow"]:

        group = detector_data[
            detector_data["speed"] == speed
        ]

        actual_seeds = set(group["seed"].tolist())

        print(
            f"{speed}: {len(group)} rows | "
            f"{len(actual_seeds)} unique seeds"
        )

        if actual_seeds != expected_seeds:
            raise ValueError(
                f"Incorrect seed coverage for {detector}, {speed}"
            )

# --------------------------------------------------
# 7. Calculate aggregate performance
# --------------------------------------------------

summary_rows = []

for (speed, detector), group in combined.groupby(
    ["speed", "detector"]
):

    tp = int(group["detected"].sum())

    fp = int(group["false_alarm_episodes"].sum())

    fn = len(group) - tp

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
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
        "detector": detector,

        "total_streams": len(group),

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
            int(group["pre_drift_false_alarm"].sum()),

        "total_pre_drift_exposure":
            total_exposure
    })

summary = pd.DataFrame(summary_rows)

# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------

detailed_path = (
    RESULTS_DIR / "speed_all_detectors_seed_results_25_seeds.csv"
)

summary_path = (
    RESULTS_DIR / "speed_all_detectors_comparison_25_seeds.csv"
)

combined.to_csv(detailed_path, index=False)
summary.to_csv(summary_path, index=False)

# --------------------------------------------------
# 9. Display final results
# --------------------------------------------------

print("\n========== FINAL EXPERIMENT COVERAGE ==========")

print("Seeds: 47–71")
print("Streams per speed: 25")
print("Speeds: 3")
print("Detectors: 5")
print("Total detector-stream evaluations:", len(combined))

print("\n========== FINAL 25-SEED COMPARISON ==========")

display_columns = [
    "speed",
    "detector",
    "total_streams",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "mean_delay_from_onset",
    "median_delay_from_onset",
    "false_alarms_per_1000_records",
    "streams_with_pre_drift_false_alarm"
]

print(
    summary[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nSaved detailed results:")
print(detailed_path)

print("\nSaved aggregate comparison:")
print(summary_path)

print("\nFinal 25-seed comparison completed successfully.")
