
import pandas as pd
import numpy as np

df = pd.read_csv(
    "results/tables/elec2_signal_diagnostics.csv"
)

# Keep only records in the monitoring phase
monitoring = df[
    df["monitoring_phase"] == "monitoring"
].copy()

# Divide monitoring records into 10 chronological blocks
monitoring["block"] = pd.qcut(
    monitoring["record"],
    q=10,
    labels=False
)

signals = [
    "rolling_error",
    "error_increase",
    "rolling_confidence",
    "confidence_drop",
    "relationship_change"
]

# Aggregate signal statistics
summary = monitoring.groupby("block").agg(
    start_record=("record", "min"),
    end_record=("record", "max"),
    total_records=("record", "count"),
    **{
        f"{signal}_mean": (signal, "mean")
        for signal in signals
    },
    **{
        f"{signal}_median": (signal, "median")
        for signal in signals
    },
    **{
        f"{signal}_std": (signal, "std")
        for signal in signals
    },
    **{
        f"{signal}_p95": (
            signal,
            lambda x: x.quantile(0.95)
        )
        for signal in signals
    }
)

print("\n========== ELEC2 SIGNAL SUMMARY BY BLOCK ==========")

print(summary.round(4).to_string())

# Save detailed summary
output_path = (
    "results/tables/elec2_signal_summary_by_block.csv"
)

summary.to_csv(output_path)

# Also produce a compact mean-only table
mean_columns = [
    "rolling_error_mean",
    "error_increase_mean",
    "rolling_confidence_mean",
    "confidence_drop_mean",
    "relationship_change_mean"
]

print("\n========== MEAN SIGNAL VALUES BY BLOCK ==========")

print(
    summary[mean_columns]
    .round(4)
    .to_string()
)

print("\nSaved summary to:", output_path)
print("\nSignal summary completed.")
