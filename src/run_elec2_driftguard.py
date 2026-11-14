
import json
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_PATH = "results/tables/elec2_signal_diagnostics.csv"

CONFIG_PATH = "results/tables/elec2_driftguard_config.json"

TIMELINE_PATH = (
    "results/tables/elec2_driftguard_monitoring.csv"
)

EPISODES_PATH = (
    "results/tables/elec2_driftguard_alert_episodes.csv"
)


# ============================================================
# 2. LOAD FROZEN CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

SIGNALS = config["signals"]
WEIGHTS = config["weights"]
PARAMETERS = config["signal_parameters"]
THRESHOLD = config["threshold"]

MONITORING_START = config["monitoring_start"]


# ============================================================
# 3. LOAD SIGNAL DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "record",
    "monitoring_phase"
] + SIGNALS

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 4. EXTRACT MONITORING PERIOD
# ============================================================

monitoring = df[
    df["monitoring_phase"] == "monitoring"
].copy()

monitoring = (
    monitoring
    .sort_values("record")
    .reset_index(drop=True)
)

if monitoring.empty:
    raise ValueError("No monitoring records found.")

if monitoring["record"].min() != MONITORING_START:
    raise ValueError(
        "Monitoring does not start at the configured record."
    )

if monitoring[SIGNALS].isna().any().any():
    raise ValueError(
        "Missing signal values during monitoring."
    )

if not np.isfinite(
    monitoring[SIGNALS].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite signal values found during monitoring."
    )

print("\n========== DRIFTGUARD MONITORING ==========")

print(
    "Monitoring records:",
    monitoring["record"].min(),
    "to",
    monitoring["record"].max()
)

print("Total monitoring observations:", len(monitoring))

print("Frozen alert threshold:", round(THRESHOLD, 5))


# ============================================================
# 5. STANDARDIZE SIGNALS
# ============================================================

for signal in SIGNALS:

    mean_value = PARAMETERS[signal]["mean"]

    std_value = PARAMETERS[signal]["std"]

    if std_value <= 0:
        raise ValueError(
            f"Invalid standard deviation for {signal}"
        )

    z_score = (
        monitoring[signal] - mean_value
    ) / std_value

    # Only positive deviations contribute to the score.
    monitoring[f"{signal}_z"] = z_score.clip(lower=0)


# ============================================================
# 6. CALCULATE HYBRID SCORE
# ============================================================

monitoring["hybrid_score"] = 0.0

for signal in SIGNALS:

    monitoring["hybrid_score"] += (
        WEIGHTS[signal] *
        monitoring[f"{signal}_z"]
    )


# ============================================================
# 7. GENERATE ALERTS
# ============================================================

monitoring["alert"] = (
    monitoring["hybrid_score"] > THRESHOLD
)

monitoring["alert"] = monitoring["alert"].astype(bool)


# ============================================================
# 8. GROUP CONSECUTIVE ALERTS INTO EPISODES
# ============================================================

alert_records = monitoring[
    monitoring["alert"]
].copy()

episodes = []

if not alert_records.empty:

    # A new episode begins whenever alert records
    # are not consecutive.
    alert_records["episode_group"] = (
        alert_records["record"].diff().ne(1).cumsum()
    )

    for episode_number, group in alert_records.groupby(
        "episode_group"
    ):

        episodes.append({

            "episode_id": len(episodes) + 1,

            "start_record": int(
                group["record"].min()
            ),

            "end_record": int(
                group["record"].max()
            ),

            "duration_records": len(group),

            "maximum_score": float(
                group["hybrid_score"].max()
            ),

            "mean_score": float(
                group["hybrid_score"].mean()
            ),

            "mean_error_z": float(
                group["error_increase_z"].mean()
            ),

            "mean_confidence_z": float(
                group["confidence_drop_z"].mean()
            ),

            "mean_relationship_z": float(
                group["relationship_change_z"].mean()
            )
        })


episode_df = pd.DataFrame(
    episodes,
    columns=[
        "episode_id",
        "start_record",
        "end_record",
        "duration_records",
        "maximum_score",
        "mean_score",
        "mean_error_z",
        "mean_confidence_z",
        "mean_relationship_z"
    ]
)


# ============================================================
# 9. SAVE MONITORING TIMELINE
# ============================================================

output_columns = [
    "record"
] + SIGNALS + [
    "error_increase_z",
    "confidence_drop_z",
    "relationship_change_z",
    "hybrid_score",
    "alert"
]

monitoring[output_columns].to_csv(
    TIMELINE_PATH,
    index=False
)


# ============================================================
# 10. SAVE ALERT EPISODES
# ============================================================

episode_df.to_csv(
    EPISODES_PATH,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

total_records = len(monitoring)

alert_count = int(
    monitoring["alert"].sum()
)

alert_percentage = (
    alert_count / total_records
) * 100

print("\n========== DRIFTGUARD RESULTS ==========")

print("Total monitored records:", total_records)

print("Alert records:", alert_count)

print(
    "Alert record percentage:",
    round(alert_percentage, 3),
    "%"
)

print("Alert episodes:", len(episode_df))

print(
    "Frozen threshold:",
    round(THRESHOLD, 5)
)

print(
    "Mean hybrid score:",
    round(monitoring["hybrid_score"].mean(), 5)
)

print(
    "Maximum hybrid score:",
    round(monitoring["hybrid_score"].max(), 5)
)

print(
    "95th percentile hybrid score:",
    round(
        monitoring["hybrid_score"].quantile(0.95),
        5
    )
)


# ============================================================
# 12. DISPLAY ALERT EPISODES
# ============================================================

print("\n========== ALERT EPISODES ==========")

if episode_df.empty:

    print("No alert episodes detected.")

else:

    print(
        episode_df.round(4).to_string(index=False)
    )


# ============================================================
# 13. SAVE CONFIRMATION
# ============================================================

print("\n========== OUTPUT FILES ==========")

print("Monitoring timeline:", TIMELINE_PATH)

print("Alert episodes:", EPISODES_PATH)

print("\nReal-world DriftGuard monitoring completed.")
