
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "multi_seed_hybrid_signal_diagnostics.csv"
)

OUTPUT_DIR = ROOT / "results" / "tables"

data = pd.read_csv(INPUT_FILE)

periods = {
    "before_drift": (900, 999),
    "drift_begins": (1000, 1099),
    "after_drift": (1100, 1199),
    "later_period": (1500, 1599)
}

signals = [
    "rolling_error",
    "rolling_confidence",
    "relationship_change"
]

results = []

for seed in sorted(data["seed"].unique()):

    seed_data = data[data["seed"] == seed]

    for period, (start, end) in periods.items():

        subset = seed_data[
            seed_data["record"].between(start, end)
        ]

        row = {
            "seed": seed,
            "period": period
        }

        for signal in signals:
            row[signal] = subset[signal].mean()

        results.append(row)

summary = pd.DataFrame(results)

# Save individual seed results
output_file = OUTPUT_DIR / "multi_seed_signal_summary.csv"
summary.to_csv(output_file, index=False)

# Calculate average and standard deviation across seeds
aggregate = (
    summary.groupby("period")[signals]
    .agg(["mean", "std"])
    .round(4)
)

print("\n========== MULTI-SEED SIGNAL SUMMARY ==========")
print(summary.round(4).to_string(index=False))

print("\n========== AVERAGE ACROSS 10 SEEDS ==========")
print(aggregate)

print("\nSaved at:", output_file)
print("Total seeds:", summary["seed"].nunique())
print("Total period comparisons:", len(summary))
