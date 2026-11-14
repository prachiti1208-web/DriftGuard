
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "tables" / "speed_tests"

SEEDS = range(47, 72)
SPEEDS = ["fast", "medium", "slow"]
DETECTORS = ["DriftGuard", "DDM", "EDDM", "ADWIN", "Page-Hinkley"]

ONSET = 800
CONFIDENCE_LAG = 100

# Fixed DriftGuard configuration.
# These values must NOT be recalibrated on held-out seeds.
ERROR_WEIGHT = 0.6
CONFIDENCE_WEIGHT = 0.1
RELATIONSHIP_WEIGHT = 0.3

ERROR_SCALE = 0.5
CONFIDENCE_SCALE = 0.05
RELATIONSHIP_SCALE = 1.2
THRESHOLD = 0.1

# --------------------------------------------------
# 1. Load diagnostic data
# --------------------------------------------------

diagnostics = pd.concat([
    pd.read_csv(
        RESULTS / "speed_test_hybrid_signal_diagnostics.csv"
    ),
    pd.read_csv(
        RESULTS / "speed_test_hybrid_signal_diagnostics_new_seeds.csv"
    )
], ignore_index=True)

# Load conventional raw alarm records.
conventional_alarms = pd.concat([
    pd.read_csv(
        RESULTS / "speed_conventional_detector_alarms.csv"
    ),
    pd.read_csv(
        RESULTS / "speed_conventional_detector_alarms_new_seeds.csv"
    )
], ignore_index=True)

# --------------------------------------------------
# 2. Identify common monitoring start
# --------------------------------------------------

start_rows = []

for speed in SPEEDS:
    for seed in SEEDS:

        data = diagnostics[
            (diagnostics["speed"] == speed) &
            (diagnostics["seed"] == seed)
        ].sort_values("record").copy()

        data["previous_confidence"] = (
            data["rolling_confidence"].shift(CONFIDENCE_LAG)
        )

        valid = (
            data["rolling_error"].notna() &
            data["relationship_change"].notna() &
            data["rolling_confidence"].notna() &
            data["previous_confidence"].notna()
        )

        valid_records = data.loc[valid, "record"]

        if valid_records.empty:
            raise ValueError(
                f"No complete-case records for {speed}, seed {seed}"
            )

        common_start = int(valid_records.min())

        if common_start >= ONSET:
            raise ValueError(
                f"Invalid common start for {speed}, seed {seed}"
            )

        exposure = ONSET - common_start

        start_rows.append({
            "speed": speed,
            "seed": seed,
            "common_start": common_start,
            "onset": ONSET,
            "common_exposure_records": exposure
        })

starts = pd.DataFrame(start_rows)

# --------------------------------------------------
# 3. Reconstruct DriftGuard alarms
# --------------------------------------------------

hybrid_rows = []

for _, start_row in starts.iterrows():

    speed = start_row["speed"]
    seed = int(start_row["seed"])
    common_start = int(start_row["common_start"])

    data = diagnostics[
        (diagnostics["speed"] == speed) &
        (diagnostics["seed"] == seed)
    ].sort_values("record").copy()

    previous_confidence = (
        data["rolling_confidence"].shift(CONFIDENCE_LAG)
    )

    error_signal = (
        (data["rolling_error"] - data["reference_error"])
        .clip(lower=0) / ERROR_SCALE
    ).clip(0, 1)

    relationship_signal = (
        data["relationship_change"] / RELATIONSHIP_SCALE
    ).clip(0, 1)

    confidence_signal = (
        (previous_confidence - data["rolling_confidence"])
        .clip(lower=0) / CONFIDENCE_SCALE
    ).clip(0, 1)

    score = (
        ERROR_WEIGHT * error_signal +
        CONFIDENCE_WEIGHT * confidence_signal +
        RELATIONSHIP_WEIGHT * relationship_signal
    )

    valid = (
        data["rolling_error"].notna() &
        data["relationship_change"].notna() &
        data["rolling_confidence"].notna() &
        previous_confidence.notna()
    )

    alarm_condition = valid & (score >= THRESHOLD)

    previous_alarm = (
        alarm_condition.shift(1).fillna(False).astype(bool)
    )

    rising_crossing = alarm_condition & (~previous_alarm)

    false_alarm_records = data.loc[
        rising_crossing &
        (data["record"] >= common_start) &
        (data["record"] < ONSET),
        "record"
    ].astype(int).tolist()

    hybrid_rows.append({
        "speed": speed,
        "seed": seed,
        "detector": "DriftGuard",
        "common_start": common_start,
        "common_exposure_records": int(ONSET - common_start),
        "false_alarm_episodes": len(false_alarm_records),
        "false_alarm_records": str(false_alarm_records),
        "pre_drift_false_alarm": int(len(false_alarm_records) > 0)
    })

hybrid_results = pd.DataFrame(hybrid_rows)

# --------------------------------------------------
# 4. Recalculate conventional detector alarms
#    inside exactly the same exposure window
# --------------------------------------------------

conventional_rows = []

for _, start_row in starts.iterrows():

    speed = start_row["speed"]
    seed = int(start_row["seed"])
    common_start = int(start_row["common_start"])

    for detector in [
        "DDM", "EDDM", "ADWIN", "Page-Hinkley"
    ]:

        alarms = conventional_alarms[
            (conventional_alarms["speed"] == speed) &
            (conventional_alarms["seed"] == seed) &
            (conventional_alarms["detector"] == detector)
        ].copy()

        alarms = alarms[
            (alarms["alarm_record"] >= common_start) &
            (alarms["alarm_record"] < ONSET)
        ].sort_values("alarm_record")

        records = (
            alarms["alarm_record"].astype(int).tolist()
        )

        # Consecutive alarm records belong to one episode.
        episode_count = 0

        if records:
            episode_count = 1

            for i in range(1, len(records)):
                if records[i] - records[i - 1] > 1:
                    episode_count += 1

        conventional_rows.append({
            "speed": speed,
            "seed": seed,
            "detector": detector,
            "common_start": common_start,
            "common_exposure_records": int(ONSET - common_start),
            "false_alarm_episodes": episode_count,
            "false_alarm_records": str(records),
            "pre_drift_false_alarm": int(episode_count > 0)
        })

conventional_results = pd.DataFrame(conventional_rows)

# --------------------------------------------------
# 5. Combine common-exposure results
# --------------------------------------------------

combined = pd.concat(
    [hybrid_results, conventional_results],
    ignore_index=True
)

combined = combined.sort_values(
    ["speed", "detector", "seed"]
).reset_index(drop=True)

# --------------------------------------------------
# 6. Aggregate results
# --------------------------------------------------

summary_rows = []

for (speed, detector), group in combined.groupby(
    ["speed", "detector"]
):

    total_fp = int(group["false_alarm_episodes"].sum())

    affected_streams = int(
        group["pre_drift_false_alarm"].sum()
    )

    total_exposure = int(
        group["common_exposure_records"].sum()
    )

    summary_rows.append({
        "speed": speed,
        "detector": detector,
        "streams": len(group),
        "false_alarm_episodes": total_fp,
        "streams_with_false_alarms": affected_streams,
        "false_alarm_stream_percentage":
            affected_streams / len(group) * 100,
        "total_common_exposure_records": total_exposure,
        "false_alarms_per_1000_records":
            total_fp / total_exposure * 1000
    })

summary = pd.DataFrame(summary_rows)

# --------------------------------------------------
# 7. Validate equal exposure
# --------------------------------------------------

print("\n========== COMMON EXPOSURE VALIDATION ==========")

print(
    starts.groupby("speed")[
        ["common_start", "common_exposure_records"]
    ].agg(["min", "max"]).to_string()
)

for speed in SPEEDS:

    group = combined[combined["speed"] == speed]

    exposure_counts = (
        group.groupby("detector")[
            "common_exposure_records"
        ].sum()
    )

    if exposure_counts.nunique() != 1:
        raise ValueError(
            f"Unequal total exposure for {speed}:\n{exposure_counts}"
        )

print("\nAll detectors have equal pre-drift exposure per speed.")

# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

combined.to_csv(
    RESULTS / "common_exposure_false_alarm_seed_results.csv",
    index=False
)

summary.to_csv(
    RESULTS / "common_exposure_false_alarm_summary.csv",
    index=False
)

starts.to_csv(
    RESULTS / "common_exposure_monitoring_windows.csv",
    index=False
)

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========== COMMON-EXPOSURE FALSE-ALARM COMPARISON ==========")

print(
    summary.round(4).to_string(index=False)
)

print("\nSaved:")
print("common_exposure_false_alarm_seed_results.csv")
print("common_exposure_false_alarm_summary.csv")
print("common_exposure_monitoring_windows.csv")

print("\nCommon-exposure evaluation completed.")
