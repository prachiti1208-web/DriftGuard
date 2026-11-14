
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "results" / "tables" / "speed_tests"
OUTPUT_DIR = ROOT / "results" / "tables" / "speed_tests"

OUTPUT_FILE = (
    OUTPUT_DIR / "speed_test_hybrid_signal_diagnostics_new_seeds.csv"
)

SEEDS = range(52, 72)
SPEEDS = ["fast", "medium", "slow"]

WINDOW_SIZE = 100

# Strictly pre-drift reference period
REFERENCE_START = 500
REFERENCE_END = 799


# -----------------------------------------
# 2. CALCULATE SIGNALS FOR ONE STREAM
# -----------------------------------------

def calculate_signals(file_path, speed, seed):

    data = pd.read_csv(file_path)

    required_columns = [
        "record",
        "feature_1",
        "feature_2",
        "actual",
        "predicted",
        "prediction_confidence"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}"
        )

    data = data.copy()

    # Prediction error
    data["error"] = (
        data["actual"] != data["predicted"]
    ).astype(int)

    data["rolling_error"] = (
        data["error"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    # Prediction confidence
    data["rolling_confidence"] = (
        data["prediction_confidence"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    # Feature-target relationships
    data["corr_f1"] = (
        data["feature_1"]
        .rolling(WINDOW_SIZE)
        .corr(data["actual"])
    )

    data["corr_f2"] = (
        data["feature_2"]
        .rolling(WINDOW_SIZE)
        .corr(data["actual"])
    )

    # Reference period
    reference = data[
        (data["record"] >= REFERENCE_START)
        & (data["record"] <= REFERENCE_END)
    ]

    if reference.empty:
        raise ValueError(
            f"Reference period missing: {file_path.name}"
        )

    reference_error = reference["rolling_error"].mean()

    reference_confidence = (
        reference["rolling_confidence"].mean()
    )

    reference_corr_f1 = reference["corr_f1"].mean()
    reference_corr_f2 = reference["corr_f2"].mean()

    # Relationship change
    data["relationship_change"] = np.sqrt(
        (data["corr_f1"] - reference_corr_f1) ** 2
        +
        (data["corr_f2"] - reference_corr_f2) ** 2
    )

    # Store reference values
    data["reference_error"] = reference_error
    data["reference_confidence"] = reference_confidence
    data["reference_corr_f1"] = reference_corr_f1
    data["reference_corr_f2"] = reference_corr_f2

    # Experiment information
    data["speed"] = speed
    data["seed"] = seed

    return data


# -----------------------------------------
# 3. PROCESS NEW STREAMS ONLY
# -----------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for speed in SPEEDS:

        print(f"\nProcessing {speed.upper()} DRIFT")

        for seed in SEEDS:

            file_path = (
                INPUT_DIR
                / f"{speed}_seed_{seed}_predictions.csv"
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Prediction file not found: {file_path}"
                )

            result = calculate_signals(
                file_path,
                speed,
                seed
            )

            all_results.append(result)

            print(
                f"Seed {seed}: "
                f"{len(result)} records processed"
            )

    # -------------------------------------
    # 4. COMBINE ALL RESULTS
    # -------------------------------------

    final_data = pd.concat(
        all_results,
        ignore_index=True
    )

    final_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------------------
    # 5. VALIDATION
    # -------------------------------------

    print("\n========== NEW SEED DIAGNOSTICS ==========")

    print("Total streams:", len(all_results))
    print("Total records:", len(final_data))

    print("\nStreams per speed:")
    print(
        final_data.groupby("speed")["seed"]
        .nunique()
    )

    print("\nRecords per speed:")
    print(
        final_data.groupby("speed")
        .size()
    )

    print("\nMissing values in signal columns:")
    print(
        final_data[
            [
                "rolling_error",
                "rolling_confidence",
                "corr_f1",
                "corr_f2",
                "relationship_change"
            ]
        ].isna().sum()
    )

    print("\nOutput saved successfully:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
