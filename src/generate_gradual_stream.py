
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------
# 1. Generate gradual concept drift
# -----------------------------------------

def generate_gradual_stream(
    seed=42,
    samples=2000,
    drift_start=800,
    drift_end=1200
):

    if not 0 <= drift_start < drift_end < samples:
        raise ValueError(
            "Require 0 <= drift_start < drift_end < samples."
        )

    rng = np.random.RandomState(seed)

    # Generate two independent features
    feature_1 = rng.normal(0, 1, samples)
    feature_2 = rng.normal(0, 1, samples)

    # Old and new concepts
    old_target = (feature_1 > 0).astype(int)
    new_target = (feature_2 > 0).astype(int)

    # Probability of selecting the new concept
    new_concept_probability = np.zeros(samples)

    transition_length = drift_end - drift_start

    for i in range(samples):

        if i < drift_start:
            new_concept_probability[i] = 0.0

        elif i >= drift_end:
            new_concept_probability[i] = 1.0

        else:
            progress = (i - drift_start) / transition_length
            new_concept_probability[i] = progress

    # Randomly select old or new concept for each observation
    use_new_concept = (
        rng.random_sample(samples)
        < new_concept_probability
    )

    target = np.where(
        use_new_concept,
        new_target,
        old_target
    )

    # Ground-truth drift indicator for evaluation only.
    # It must never be supplied to the detector.
    drift = (
        np.arange(samples) >= drift_start
    ).astype(int)

    data = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "target": target,
        "drift": drift
    })

    return data


# -----------------------------------------
# 2. Command-line arguments
# -----------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate gradual concept drift data."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--drift-start",
        type=int,
        default=800
    )

    parser.add_argument(
        "--drift-end",
        type=int,
        default=1200
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/streams/gradual_seed_42.csv"
    )

    args = parser.parse_args()

    # Generate dataset
    data = generate_gradual_stream(
        seed=args.seed,
        samples=args.samples,
        drift_start=args.drift_start,
        drift_end=args.drift_end
    )

    # Save dataset
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(output_path, index=False)

    print("\nGradual stream generated successfully!")

    print(f"Seed: {args.seed}")
    print(f"Observations: {args.samples}")
    print(f"Drift starts: {args.drift_start}")
    print(f"Drift ends: {args.drift_end}")

    print(f"\nSaved to: {output_path}")

    print("\nDataset shape:")
    print(data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nTarget distribution:")
    print(data["target"].value_counts().sort_index())

    print("\nDrift indicator distribution:")
    print(data["drift"].value_counts().sort_index())


if __name__ == "__main__":
    main()
