
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results" / "tables" / "speed_tests"
FIGURES = ROOT / "results" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# Load experimental results
delays = pd.read_csv(
    RESULTS / "speed_all_detectors_seed_results_25_seeds.csv"
)

false_alarms = pd.read_csv(
    RESULTS / "common_exposure_false_alarm_summary.csv"
)

statistics = pd.read_csv(
    RESULTS / "statistical_comparison_25_seeds.csv"
)

speeds = ["fast", "medium", "slow"]

detectors = [
    "DriftGuard",
    "DDM",
    "EDDM",
    "ADWIN",
    "Page-Hinkley"
]

# --------------------------------------------------
# FIGURE 1: Detection delay across drift speeds
# --------------------------------------------------

summary = (
    delays.groupby(["speed", "detector"])
    ["delay_from_onset"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(10, 6))

for detector in detectors:

    subset = summary[
        summary["detector"] == detector
    ]

    subset = (
        subset.set_index("speed")
        .reindex(speeds)
        .reset_index()
    )

    plt.errorbar(
        speeds,
        subset["mean"],
        yerr=subset["std"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=detector
    )

plt.xlabel("Gradual drift speed")
plt.ylabel("Detection delay from drift onset (records)")
plt.title("Detection Delay Across Different Drift Speeds")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "figure_1_detection_delay.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# FIGURE 2: Common-exposure false alarms
# --------------------------------------------------

plt.figure(figsize=(11, 6))

x = np.arange(len(speeds))
width = 0.15

for i, detector in enumerate(detectors):

    subset = false_alarms[
        false_alarms["detector"] == detector
    ]

    subset = (
        subset.set_index("speed")
        .reindex(speeds)
        .reset_index()
    )

    positions = (
        x + (i - (len(detectors) - 1) / 2) * width
    )

    plt.bar(
        positions,
        subset["false_alarm_episodes"],
        width,
        label=detector
    )

plt.xticks(x, speeds)
plt.xlabel("Gradual drift speed")
plt.ylabel("False-alarm episodes")
plt.title(
    "False Alarms Under Equal Pre-Drift Monitoring Exposure"
)

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "figure_2_false_alarms.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# FIGURE 3: Paired delay differences with 95% CI
# --------------------------------------------------

plot_data = statistics.copy()

plot_data["label"] = (
    plot_data["speed"].str.capitalize()
    + " — "
    + plot_data["comparison"].str.replace(
        "DriftGuard_vs_", "", regex=False
    )
)

plot_data = plot_data.iloc[::-1].reset_index(drop=True)

y = np.arange(len(plot_data))

lower_error = (
    plot_data["mean_paired_difference"]
    - plot_data["bootstrap_95_ci_lower"]
)

upper_error = (
    plot_data["bootstrap_95_ci_upper"]
    - plot_data["mean_paired_difference"]
)

plt.figure(figsize=(11, 8))

plt.errorbar(
    plot_data["mean_paired_difference"],
    y,
    xerr=[
        lower_error,
        upper_error
    ],
    fmt="o",
    capsize=4
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=1
)

plt.yticks(y, plot_data["label"])

plt.xlabel(
    "Paired delay difference (DriftGuard - comparator)"
)

plt.title(
    "Paired Detection Delay Differences with 95% Bootstrap CIs"
)

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES / "figure_3_statistical_comparison.png",
    dpi=300
)

plt.close()

print("\n========== FIGURES GENERATED ==========")

for filename in [
    "figure_1_detection_delay.png",
    "figure_2_false_alarms.png",
    "figure_3_statistical_comparison.png"
]:

    print(FIGURES / filename)

print("\nAll three research figures generated successfully.")
