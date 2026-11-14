
import pandas as pd

df = pd.read_csv(
    "results/tables/hybrid_variant_comparison.csv"
)

TRUE_DRIFT_POINT = 1000
THRESHOLD = 0.30

# Shared signal weights in the three-signal model
error_weight = 0.40
relationship_weight = 0.35
confidence_weight = 0.25

# Normalize the two shared weights so they sum to 1
shared_total = error_weight + relationship_weight

two_signal_score = (
    (error_weight / shared_total) * df["error_signal"]
    + (relationship_weight / shared_total)
    * df["relationship_signal"]
)

three_signal_score = (
    error_weight * df["error_signal"]
    + confidence_weight * df["confidence_signal"]
    + relationship_weight * df["relationship_signal"]
)

df["controlled_two_signal_score"] = two_signal_score
df["controlled_three_signal_score"] = three_signal_score


def evaluate(score_column, name):

    alarms = df.loc[
        df[score_column] >= THRESHOLD,
        "record"
    ].tolist()

    false_alarms = [
        p for p in alarms if p < TRUE_DRIFT_POINT
    ]

    valid_alarms = [
        p for p in alarms if p >= TRUE_DRIFT_POINT
    ]

    print(f"\n========== {name} ==========")
    print("False alarm count:", len(false_alarms))

    if valid_alarms:
        first = valid_alarms[0]
        print("First detection:", first)
        print("Detection delay:", first - TRUE_DRIFT_POINT)
    else:
        print("No detection.")

evaluate(
    "controlled_two_signal_score",
    "CONTROLLED TWO-SIGNAL MODEL"
)

evaluate(
    "controlled_three_signal_score",
    "CONTROLLED THREE-SIGNAL MODEL"
)

df.to_csv(
    "results/tables/confidence_ablation_results.csv",
    index=False
)

print("\nControlled ablation results saved!")
