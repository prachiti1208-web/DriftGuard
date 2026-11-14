
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "tables" / "speed_tests"

INPUT_FILE = (
    RESULTS / "speed_all_detectors_seed_results_25_seeds.csv"
)

OUTPUT_FILE = (
    RESULTS / "statistical_comparison_25_seeds.csv"
)

SEEDS = list(range(47, 72))
SPEEDS = ["fast", "medium", "slow"]

BASELINE = "DriftGuard"

COMPARATORS = [
    "DDM",
    "EDDM",
    "ADWIN",
    "Page-Hinkley"
]

N_BOOTSTRAP = 10000
RANDOM_SEED = 42

# --------------------------------------------------
# 1. Load results
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "speed",
    "seed",
    "detector",
    "delay_from_onset",
    "detected"
}

missing = required_columns - set(data.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Keep only successfully detected streams.
data = data[data["detected"] == 1].copy()

# Validate complete seed coverage.
for speed in SPEEDS:
    for detector in [BASELINE] + COMPARATORS:

        subset = data[
            (data["speed"] == speed) &
            (data["detector"] == detector)
        ]

        if set(subset["seed"]) != set(SEEDS):
            raise ValueError(
                f"Incomplete seed coverage: {speed}, {detector}"
            )

print("Input validation successful.")
print("25 paired seeds per detector and speed.")

# --------------------------------------------------
# 2. Compare paired detection delays
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

results = []

for speed in SPEEDS:

    hybrid = data[
        (data["speed"] == speed) &
        (data["detector"] == BASELINE)
    ][["seed", "delay_from_onset"]].rename(
        columns={"delay_from_onset": "driftguard_delay"}
    )

    for detector in COMPARATORS:

        conventional = data[
            (data["speed"] == speed) &
            (data["detector"] == detector)
        ][["seed", "delay_from_onset"]].rename(
            columns={"delay_from_onset": "comparison_delay"}
        )

        paired = hybrid.merge(
            conventional,
            on="seed",
            validate="one_to_one"
        ).sort_values("seed")

        driftguard_delay = paired[
            "driftguard_delay"
        ].to_numpy()

        conventional_delay = paired[
            "comparison_delay"
        ].to_numpy()

        # Negative = DriftGuard is faster.
        differences = (
            driftguard_delay - conventional_delay
        )

        mean_difference = float(np.mean(differences))
        median_difference = float(np.median(differences))

        # Count paired outcomes.
        wins = int(np.sum(differences < 0))
        ties = int(np.sum(differences == 0))
        losses = int(np.sum(differences > 0))

        # Paired Wilcoxon signed-rank test.
        if np.all(differences == 0):
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )

            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        # Paired bootstrap confidence interval.
        bootstrap_means = np.empty(N_BOOTSTRAP)

        for i in range(N_BOOTSTRAP):

            sample = rng.choice(
                differences,
                size=len(differences),
                replace=True
            )

            bootstrap_means[i] = np.mean(sample)

        ci_lower, ci_upper = np.percentile(
            bootstrap_means,
            [2.5, 97.5]
        )

        results.append({

            "speed": speed,
            "comparison": f"DriftGuard_vs_{detector}",

            "n_seeds": len(differences),

            "driftguard_mean_delay":
                float(np.mean(driftguard_delay)),

            "conventional_mean_delay":
                float(np.mean(conventional_delay)),

            "mean_paired_difference":
                mean_difference,

            "median_paired_difference":
                median_difference,

            "bootstrap_95_ci_lower":
                float(ci_lower),

            "bootstrap_95_ci_upper":
                float(ci_upper),

            "driftguard_faster": wins,
            "equal_delay": ties,
            "driftguard_slower": losses,

            "wilcoxon_statistic": statistic,
            "p_value": p_value
        })

summary = pd.DataFrame(results)

# --------------------------------------------------
# 3. Holm correction for 12 comparisons
# --------------------------------------------------

p_values = summary["p_value"].to_numpy()

order = np.argsort(p_values)
adjusted = np.empty(len(p_values))

running_max = 0.0
m = len(p_values)

for rank, index in enumerate(order):

    corrected = (m - rank) * p_values[index]

    running_max = max(running_max, corrected)

    adjusted[index] = min(running_max, 1.0)

summary["holm_adjusted_p"] = adjusted

summary["significant_after_holm"] = (
    summary["holm_adjusted_p"] < 0.05
)

# --------------------------------------------------
# 4. Add interpretation
# --------------------------------------------------

summary["interpretation"] = np.where(
    summary["mean_paired_difference"] < 0,
    "DriftGuard faster on average",
    np.where(
        summary["mean_paired_difference"] > 0,
        "Conventional detector faster on average",
        "Equal mean delay"
    )
)

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

summary.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

display_columns = [
    "speed",
    "comparison",
    "mean_paired_difference",
    "median_paired_difference",
    "bootstrap_95_ci_lower",
    "bootstrap_95_ci_upper",
    "driftguard_faster",
    "equal_delay",
    "driftguard_slower",
    "p_value",
    "holm_adjusted_p",
    "significant_after_holm"
]

print("\n========== PAIRED STATISTICAL COMPARISON ==========")

print(
    summary[display_columns]
    .round(5)
    .to_string(index=False)
)

print("\n========== INTERPRETATION ==========")

for _, row in summary.iterrows():

    print(
        f"\n{row['speed'].upper()} | {row['comparison']}"
    )

    print(
        f"Mean paired delay difference: "
        f"{row['mean_paired_difference']:.2f} records"
    )

    print(
        f"95% bootstrap CI: "
        f"[{row['bootstrap_95_ci_lower']:.2f}, "
        f"{row['bootstrap_95_ci_upper']:.2f}]"
    )

    print(
        f"Holm-adjusted p-value: "
        f"{row['holm_adjusted_p']:.5f}"
    )

    print(
        "Statistically significant:",
        row["significant_after_holm"]
    )

print("\nSaved:", OUTPUT_FILE)
print("\nStatistical comparison completed.")
