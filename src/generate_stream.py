
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------
# 1. Configuration
# -----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--samples", type=int, default=2000)
parser.add_argument("--drift-point", type=int, default=1000)
parser.add_argument(
    "--output",
    type=str,
    default="data/processed/abrupt_drift_stream.csv"
)

args = parser.parse_args()

RANDOM_STATE = args.seed
N_SAMPLES = args.samples
DRIFT_POINT = args.drift_point

if not 0 < DRIFT_POINT < N_SAMPLES:
    raise ValueError("Drift point must be between 0 and sample count.")

# RandomState preserves the original generator's behaviour for seed 42
rng = np.random.RandomState(RANDOM_STATE)

# -----------------------------------
# 2. Generate input features
# -----------------------------------

feature_1 = rng.randn(N_SAMPLES)
feature_2 = rng.randn(N_SAMPLES)

# -----------------------------------
# 3. Generate target labels
# -----------------------------------

target = []

for i in range(N_SAMPLES):

    if i < DRIFT_POINT:
        score = feature_1[i]
    else:
        score = feature_2[i]

    label = 1 if score > 0 else 0
    target.append(label)

# -----------------------------------
# 4. Create dataset
# -----------------------------------

data = pd.DataFrame({
    "feature_1": feature_1,
    "feature_2": feature_2,
    "target": target
})

data["drift"] = 0
data.loc[DRIFT_POINT:, "drift"] = 1

# -----------------------------------
# 5. Save dataset
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

output_path = ROOT / args.output
output_path.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output_path, index=False)

# -----------------------------------
# 6. Display information
# -----------------------------------

print("================================")
print("DRIFTGUARD DATA GENERATOR")
print("================================")

print("Dataset generated successfully!")
print("Random seed:", RANDOM_STATE)
print("Total records:", len(data))
print("Total features:", 2)
print("True drift point:", DRIFT_POINT)
print("Dataset shape:", data.shape)

print("\nTarget distribution:")
print(data["target"].value_counts())

print("\nDrift distribution:")
print(data["drift"].value_counts())

print("\nSaved at:", output_path)
