
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------
# 1. Configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR = ROOT / "results" / "tables"

WINDOW_SIZE = 100
SEEDS = range(42, 52)

# Gradual drift begins at record 800
# Use only pre-drift records for reference
REFERENCE_START = 500
REFERENCE_END = 799

all_results = []

# -----------------------------------
# 2. Process each seed
# -----------------------------------

for seed in SEEDS:

    print(f"\nAnalyzing gradual-drift seed {seed}...")

    input_file = (
        INPUT_DIR / f"gradual_seed_{seed}_predictions.csv"
    )

    data = pd.read_csv(input_file)

    # Prediction error
    data["error"] = (
        data["actual"] != data["predicted"]
    ).astype(int)

    # Rolling error and confidence
    data["rolling_error"] = (
        data["error"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    data["rolling_confidence"] = (
        data["prediction_confidence"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    # Rolling feature-target correlations
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

    # -----------------------------------
    # 3. Reference period
    # -----------------------------------

    reference = data[
        data["record"].between(
            REFERENCE_START,
            REFERENCE_END
        )
    ]

    ref_error = reference["error"].mean()

    ref_confidence = (
        reference["prediction_confidence"].mean()
    )

    ref_corr_f1 = (
        reference["feature_1"].corr(reference["actual"])
    )

    ref_corr_f2 = (
        reference["feature_2"].corr(reference["actual"])
    )

    # -----------------------------------
    # 4. Relationship change
    # -----------------------------------

    data["relationship_change"] = np.sqrt(
        (data["corr_f1"] - ref_corr_f1) ** 2
        +
        (data["corr_f2"] - ref_corr_f2) ** 2
    )

    # Store reference values
    data["reference_error"] = ref_error
    data["reference_confidence"] = ref_confidence
    data["reference_corr_f1"] = ref_corr_f1
    data["reference_corr_f2"] = ref_corr_f2

    # Add seed identifier
    data["seed"] = seed

    all_results.append(data)

    print("Reference error:", round(ref_error, 4))
    print("Reference confidence:", round(ref_confidence, 4))
    print("Reference corr f1:", round(ref_corr_f1, 4))
    print("Reference corr f2:", round(ref_corr_f2, 4))

# -----------------------------------
# 5. Combine all seeds
# -----------------------------------

combined = pd.concat(
    all_results,
    ignore_index=True
)

# -----------------------------------
# 6. Save diagnostics
# -----------------------------------

output_file = (
    OUTPUT_DIR /
    "gradual_hybrid_signal_diagnostics_v2.csv"
)

combined.to_csv(output_file, index=False)

print("\n" + "=" * 50)
print("GRADUAL-DRIFT SIGNAL ANALYSIS COMPLETED")
print("=" * 50)

print("Total records:", len(combined))
print("Total seeds:", combined["seed"].nunique())
print("Reference period:", REFERENCE_START, "-", REFERENCE_END)

print("\nSaved at:", output_file)
