
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT / "results" / "tables" / "speed_tests"

HYBRID_FILE = RESULT_DIR / "speed_robustness_summary.csv"

CONVENTIONAL_FILE = (
    RESULT_DIR / "speed_conventional_summary.csv"
)

OUTPUT_FILE = RESULT_DIR / "speed_all_detectors_comparison.csv"


# --------------------------------------------------
# 2. Load experiment summaries
# --------------------------------------------------

def main():

    print("\n========== BUILDING SPEED COMPARISON TABLE ==========")

    if not HYBRID_FILE.exists():
        raise FileNotFoundError(HYBRID_FILE)

    if not CONVENTIONAL_FILE.exists():
        raise FileNotFoundError(CONVENTIONAL_FILE)

    hybrid = pd.read_csv(HYBRID_FILE)

    conventional = pd.read_csv(CONVENTIONAL_FILE)

    print("\nHybrid summary loaded:", len(hybrid), "rows")
    print("Conventional summary loaded:", len(conventional), "rows")


    # --------------------------------------------------
    # 3. Standardize hybrid results
    # --------------------------------------------------

    hybrid["detector"] = "DriftGuard (Three-signal)"

    hybrid_columns = [
        "speed",
        "detector",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "F1",
        "mean_delay_from_onset",
        "median_delay_from_onset",
        "mean_delay_from_midpoint",
        "mean_delay_from_completion",
        "false_alarms_per_1000_records",
        "streams_with_pre_drift_false_alarm"
    ]

    hybrid = hybrid[hybrid_columns].copy()


    # --------------------------------------------------
    # 4. Standardize conventional detector results
    # --------------------------------------------------

    conventional_columns = [
        "speed",
        "detector",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "F1",
        "mean_delay_from_onset",
        "median_delay_from_onset",
        "mean_delay_from_midpoint",
        "mean_delay_from_completion",
        "false_alarms_per_1000_records",
        "streams_with_pre_drift_false_alarm"
    ]

    missing = set(conventional_columns) - set(conventional.columns)

    if missing:
        raise ValueError(
            f"Missing conventional summary columns: {missing}"
        )

    conventional = conventional[conventional_columns].copy()


    # --------------------------------------------------
    # 5. Combine results
    # --------------------------------------------------

    comparison = pd.concat(
        [hybrid, conventional],
        ignore_index=True
    )

    speed_order = {
        "fast": 0,
        "medium": 1,
        "slow": 2
    }

    detector_order = {
        "DriftGuard (Three-signal)": 0,
        "DDM": 1,
        "EDDM": 2,
        "ADWIN": 3,
        "Page-Hinkley": 4
    }

    comparison["speed_order"] = (
        comparison["speed"].map(speed_order)
    )

    comparison["detector_order"] = (
        comparison["detector"].map(detector_order)
    )

    comparison = comparison.sort_values(
        ["speed_order", "detector_order"]
    )

    comparison = comparison.drop(
        columns=["speed_order", "detector_order"]
    ).reset_index(drop=True)


    # --------------------------------------------------
    # 6. Round numerical values
    # --------------------------------------------------

    numeric_columns = comparison.select_dtypes(
        include="number"
    ).columns

    comparison[numeric_columns] = comparison[
        numeric_columns
    ].round(4)


    # --------------------------------------------------
    # 7. Save consolidated table
    # --------------------------------------------------

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== CONSOLIDATED COMPARISON ==========")

    print(
        comparison.to_string(index=False)
    )

    print("\nTotal comparison rows:", len(comparison))

    print("\nDetectors:")
    print(comparison["detector"].unique())

    print("\nDrift speeds:")
    print(comparison["speed"].unique())

    print("\nSaved consolidated table:")
    print(OUTPUT_FILE)

    print("\nComparison table created successfully!")


if __name__ == "__main__":
    main()
