
import pandas as pd
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIAGNOSTICS_FILE = (
    ROOT / "results" / "tables" / "speed_tests"
    / "speed_test_hybrid_signal_diagnostics_new_seeds.csv"
)

RESULTS_FILE = (
    ROOT / "results" / "tables" / "speed_tests"
    / "speed_robustness_new_seeds_seed_results.csv"
)

# Fixed configuration: do not recalibrate on test seeds.
W_ERROR = 0.6
W_CONFIDENCE = 0.1
W_RELATIONSHIP = 0.3

ERROR_SCALE = 0.5
RELATIONSHIP_SCALE = 1.2
CONFIDENCE_SCALE = 0.05
CONFIDENCE_LAG = 100

THRESHOLD = 0.1
PROBLEM_SEEDS = [57, 64, 66, 70]

diagnostics = pd.read_csv(DIAGNOSTICS_FILE)
results = pd.read_csv(RESULTS_FILE)

# Use fast speed because pre-drift data is identical across speeds.
data = diagnostics[diagnostics["speed"] == "fast"].copy()
data = data.sort_values(["seed", "record"])

print("\n========== FALSE ALARM SCORE INVESTIGATION ==========")

for seed in PROBLEM_SEEDS:

    group = data[data["seed"] == seed].copy()

    # Reference error from records 500–799.
    reference_period = group[
        group["record"].between(500, 799)
    ]

    reference_error = reference_period[
        "rolling_error"
    ].mean()

    # Reconstruct the three detector signals.
    group["error_signal"] = (
        (group["rolling_error"] - reference_error)
        .clip(lower=0) / ERROR_SCALE
    ).clip(0, 1)

    group["relationship_signal"] = (
        group["relationship_change"]
        / RELATIONSHIP_SCALE
    ).clip(0, 1)

    group["confidence_signal"] = (
        group["rolling_confidence"].shift(CONFIDENCE_LAG)
        - group["rolling_confidence"]
    ).clip(lower=0).div(CONFIDENCE_SCALE).clip(0, 1)

    # Weighted contributions.
    group["error_contribution"] = (
        W_ERROR * group["error_signal"]
    )

    group["confidence_contribution"] = (
        W_CONFIDENCE * group["confidence_signal"]
    )

    group["relationship_contribution"] = (
        W_RELATIONSHIP * group["relationship_signal"]
    )

    group["hybrid_score"] = (
        group["error_contribution"]
        + group["confidence_contribution"]
        + group["relationship_contribution"]
    )

    result_row = results[
        (results["speed"] == "fast")
        & (results["seed"] == seed)
    ].iloc[0]

    alarms = ast.literal_eval(result_row["alarms"])
    false_alarms = [record for record in alarms if record < 800]

    print(f"\n\n========== SEED {seed} ==========")
    print(f"Reference error: {reference_error:.4f}")

    for alarm_record in false_alarms:

        row = group[group["record"] == alarm_record]

        if row.empty:
            continue

        row = row.iloc[0]

        print(f"\nFalse alarm at record: {alarm_record}")
        print(f"Hybrid score: {row['hybrid_score']:.4f}")
        print(f"Threshold: {THRESHOLD}")

        print(
            f"Error signal: {row['error_signal']:.4f} "
            f"| contribution: {row['error_contribution']:.4f}"
        )

        print(
            f"Confidence signal: {row['confidence_signal']:.4f} "
            f"| contribution: {row['confidence_contribution']:.4f}"
        )

        print(
            f"Relationship signal: {row['relationship_signal']:.4f} "
            f"| contribution: {row['relationship_contribution']:.4f}"
        )

        print("\nNearby records:")

        nearby = group[
            group["record"].between(
                alarm_record - 2,
                alarm_record + 2
            )
        ]

        print(
            nearby[
                [
                    "record",
                    "rolling_error",
                    "rolling_confidence",
                    "relationship_change",
                    "error_signal",
                    "confidence_signal",
                    "relationship_signal",
                    "hybrid_score"
                ]
            ].round(4).to_string(index=False)
        )

print("\n========== INVESTIGATION COMPLETE ==========")
