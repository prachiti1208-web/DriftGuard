
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "tables" / "speed_tests"
    / "speed_robustness_new_seeds_seed_results.csv"
)

data = pd.read_csv(INPUT_FILE)

print("\n========== PRE-DRIFT FALSE ALARM INVESTIGATION ==========")

problem_streams = data[
    data["pre_drift_false_alarms"] > 0
]

for _, row in problem_streams.iterrows():

    alarms = eval(row["alarms"]) if isinstance(
        row["alarms"], str
    ) else row["alarms"]

    pre_alarms = [
        alarm for alarm in alarms if alarm < 800
    ]

    print(
        f"\nSpeed: {row['speed']} | Seed: {row['seed']}"
    )

    print("False-alarm count:", len(pre_alarms))
    print("False-alarm records:", pre_alarms)

print("\n========== UNIQUE PROBLEMATIC SEEDS ==========")

print(
    sorted(problem_streams["seed"].unique().tolist())
)
