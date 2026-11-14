
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "tables"

HYBRID_FILE = TABLE_DIR / "gradual_hybrid_test_summary.csv"
CONVENTIONAL_FILE = TABLE_DIR / "gradual_detector_test_summary.csv"

hybrid = pd.read_csv(HYBRID_FILE)
conventional = pd.read_csv(CONVENTIONAL_FILE)

# Standardize detector name column
hybrid = hybrid.rename(columns={"variant": "detector"})

# Combine both experiment summaries
comparison = pd.concat(
    [conventional, hybrid],
    ignore_index=True
)

# Keep the most important research metrics
columns = [
    "detector",
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

comparison = comparison[columns]

# Sort by mean delay
comparison = comparison.sort_values(
    "mean_delay_from_onset"
).reset_index(drop=True)

output_file = TABLE_DIR / "gradual_all_detectors_comparison.csv"

comparison.to_csv(output_file, index=False)

print("\n========== FINAL GRADUAL DRIFT COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

print("\nSaved comparison:")
print(output_file)
