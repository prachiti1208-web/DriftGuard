import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

monitoring_path = ROOT / "results/tables/elec2_driftguard_monitoring.csv"
diagnostics_path = ROOT / "results/tables/elec2_signal_diagnostics.csv"

monitoring = pd.read_csv(monitoring_path)
diagnostics = pd.read_csv(diagnostics_path)

# Merge actual prediction correctness
df = monitoring.merge(
    diagnostics[["record", "correct"]],
    on="record",
    how="left"
)

df = df.sort_values("record").reset_index(drop=True)

if df["correct"].isna().any():
    raise ValueError("Some monitoring records have no matching correctness value.")

df["correct"] = df["correct"].astype(int)
df["alert"] = df["alert"].astype(bool)

HORIZON = 200

# Use non-overlapping 200-record future windows.
results = []

for start in range(0, len(df) - HORIZON, HORIZON):
    current = df.iloc[start]
    future = df.iloc[start + 1:start + 1 + HORIZON]

    results.append({
        "start_record": int(current["record"]),
        "end_record": int(future["record"].iloc[-1]),
        "alert_at_start": bool(current["alert"]),
        "starting_score": float(current["hybrid_score"]),
        "future_accuracy": future["correct"].mean() * 100
    })

windows = pd.DataFrame(results)

summary = (
    windows.groupby("alert_at_start")
    .agg(
        windows=("future_accuracy", "count"),
        mean_future_accuracy=("future_accuracy", "mean"),
        median_future_accuracy=("future_accuracy", "median"),
        mean_starting_score=("starting_score", "mean")
    )
    .reset_index()
)

print("\n========== FUTURE PERFORMANCE VALIDATION ==========")
print(summary.to_string(index=False))

alert_windows = windows[windows["alert_at_start"] == True]
normal_windows = windows[windows["alert_at_start"] == False]

if len(alert_windows) and len(normal_windows):
    difference = (
        normal_windows["future_accuracy"].mean()
        - alert_windows["future_accuracy"].mean()
    )

    print("\n========== COMPARISON ==========")
    print(f"Alert-start windows: {len(alert_windows)}")
    print(f"Non-alert-start windows: {len(normal_windows)}")
    print(f"Future accuracy after alert: "
          f"{alert_windows['future_accuracy'].mean():.2f}%")
    print(f"Future accuracy after no alert: "
          f"{normal_windows['future_accuracy'].mean():.2f}%")
    print(f"Difference (no alert minus alert): {difference:.2f} percentage points")

output_path = ROOT / "results/tables/elec2_early_warning_validation.csv"
windows.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")