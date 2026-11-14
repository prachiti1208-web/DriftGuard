
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/elec2_real_world.csv")

# Divide the chronological stream into 10 equal-sized blocks
df["block"] = pd.qcut(
    df["record"],
    q=10,
    labels=False
)

print("\n========== TEMPORAL BLOCK ANALYSIS ==========")

# Target distribution over time
target_summary = df.groupby("block").agg(
    start_record=("record", "min"),
    end_record=("record", "max"),
    total_records=("target", "size"),
    true_percentage=("target", lambda x: x.mean() * 100)
)

print("\nTARGET DISTRIBUTION BY BLOCK")
print(target_summary.round(2).to_string())

# Feature means over time
features = [
    "date",
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer"
]

feature_summary = df.groupby("block")[features].mean()

print("\nFEATURE MEANS BY BLOCK")
print(feature_summary.round(4).to_string())

# Feature-target correlation over time
correlation_rows = []

for block, group in df.groupby("block"):
    row = {"block": block}

    for feature in features:
        row[feature] = group[feature].corr(
            group["target"].astype(int)
        )

    correlation_rows.append(row)

correlation_df = pd.DataFrame(correlation_rows)

print("\nFEATURE-TARGET CORRELATIONS BY BLOCK")
print(correlation_df.round(4).to_string(index=False))

# Save results
target_summary.to_csv(
    "results/tables/elec2_target_by_block.csv"
)

feature_summary.to_csv(
    "results/tables/elec2_feature_means_by_block.csv"
)

correlation_df.to_csv(
    "results/tables/elec2_correlations_by_block.csv",
    index=False
)

print("\nTemporal analysis completed.")
