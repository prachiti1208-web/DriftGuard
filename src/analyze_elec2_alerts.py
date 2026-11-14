
import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD MONITORING RESULTS
# ============================================================

PATH = "results/tables/elec2_driftguard_monitoring.csv"

df = pd.read_csv(PATH)

diagnostics = pd.read_csv(
    "results/tables/elec2_signal_diagnostics.csv"
)

df = df.merge(
    diagnostics[["record", "correct"]],
    on="record",
    how="left",
    validate="one_to_one"
)

df = df.sort_values("record").reset_index(drop=True)

if df["correct"].isna().any():
    raise ValueError("Missing prediction correctness values.")

df["alert"] = df["alert"].astype(bool)

df["correct"] = df["correct"].astype(int)


# ============================================================
# 2. CREATE TEN CHRONOLOGICAL BLOCKS
# ============================================================

df["block"] = pd.qcut(
    df["record"].rank(method="first"),
    q=10,
    labels=False
)


# ============================================================
# 3. BLOCK-WISE PERFORMANCE AND ALERT ANALYSIS
# ============================================================

summary = (
    df.groupby("block")
    .agg(
        start_record=("record", "min"),
        end_record=("record", "max"),
        total_records=("record", "count"),

        accuracy=("correct", "mean"),

        alert_records=("alert", "sum"),

        mean_hybrid_score=("hybrid_score", "mean"),

        max_hybrid_score=("hybrid_score", "max"),

        mean_error_increase=("error_increase", "mean"),

        mean_confidence_drop=("confidence_drop", "mean"),

        mean_relationship_change=(
            "relationship_change", "mean"
        )
    )
    .reset_index()
)


summary["alert_percentage"] = (
    summary["alert_records"] /
    summary["total_records"]
) * 100


summary["accuracy"] = summary["accuracy"] * 100


# ============================================================
# 4. PRINT RESULTS
# ============================================================

print("\n========== CHRONOLOGICAL ALERT ANALYSIS ==========")

print(
    summary.round(4).to_string(index=False)
)


# ============================================================
# 5. OVERALL PERFORMANCE
# ============================================================

overall_accuracy = df["correct"].mean() * 100

overall_alert_percentage = (
    df["alert"].mean() * 100
)

print("\n========== OVERALL RESULTS ==========")

print(
    "Overall prequential accuracy:",
    round(overall_accuracy, 3),
    "%"
)

print(
    "Overall alert percentage:",
    round(overall_alert_percentage, 3),
    "%"
)

print(
    "Total records:",
    len(df)
)


# ============================================================
# 6. COMPARE ALERT AND NON-ALERT RECORDS
# ============================================================

print("\n========== ALERT VS NON-ALERT PERFORMANCE ==========")

comparison = (
    df.groupby("alert")
    .agg(
        records=("record", "count"),
        accuracy=("correct", "mean"),
        mean_score=("hybrid_score", "mean"),
        mean_error_increase=("error_increase", "mean"),
        mean_confidence_drop=("confidence_drop", "mean"),
        mean_relationship_change=(
            "relationship_change", "mean"
        )
    )
)

comparison["accuracy"] *= 100

print(comparison.round(4).to_string())


# ============================================================
# 7. SAVE SUMMARY
# ============================================================

OUTPUT_PATH = (
    "results/tables/elec2_alert_block_analysis.csv"
)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved block analysis to:", OUTPUT_PATH)

print("\nAlert analysis completed.")
