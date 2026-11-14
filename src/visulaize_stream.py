
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "processed" / "abrupt_drift_stream.csv"
data = pd.read_csv(data_path)

# Separate data before and after drift
before = data[data["drift"] == 0]
after = data[data["drift"] == 1]

# Calculate correlations
print("CORRELATION BEFORE DRIFT")
print(before[["feature_1", "feature_2"]].corrwith(before["target"]))

print("\nCORRELATION AFTER DRIFT")
print(after[["feature_1", "feature_2"]].corrwith(after["target"]))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Before drift: feature 1
axes[0, 0].scatter(
    before["feature_1"],
    before["target"],
    alpha=0.3
)
axes[0, 0].set_title("Before Drift: Feature 1 vs Target")

# Before drift: feature 2
axes[0, 1].scatter(
    before["feature_2"],
    before["target"],
    alpha=0.3
)
axes[0, 1].set_title("Before Drift: Feature 2 vs Target")

# After drift: feature 1
axes[1, 0].scatter(
    after["feature_1"],
    after["target"],
    alpha=0.3
)
axes[1, 0].set_title("After Drift: Feature 1 vs Target")

# After drift: feature 2
axes[1, 1].scatter(
    after["feature_2"],
    after["target"],
    alpha=0.3
)
axes[1, 1].set_title("After Drift: Feature 2 vs Target")

for ax in axes.flat:
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Target")
    ax.grid(alpha=0.2)

plt.tight_layout()

# Save figure
output_path = project_root / "results" / "figures" / "concept_drift.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nVisualization saved to:", output_path)
