
import pandas as pd
import matplotlib.pyplot as plt

# Load diagnostic data
df = pd.read_csv(
    "results/tables/hybrid_signal_diagnostics.csv"
)

DRIFT_POINT = 1000

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

# Signal 1: Rolling error
axes[0].plot(
    df["record"],
    df["rolling_error"],
    label="Rolling Error Rate"
)
axes[0].set_title("Signal 1: Prediction Error")
axes[0].set_ylabel("Error Rate")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Signal 2: Rolling confidence
axes[1].plot(
    df["record"],
    df["rolling_confidence"],
    label="Rolling Prediction Confidence"
)
axes[1].set_title("Signal 2: Prediction Confidence")
axes[1].set_ylabel("Confidence")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Signal 3: Relationship change
axes[2].plot(
    df["record"],
    df["relationship_change"],
    label="Feature-Target Relationship Change"
)
axes[2].set_title("Signal 3: Feature-Target Relationship Change")
axes[2].set_ylabel("Relationship Change")
axes[2].set_xlabel("Record Number")
axes[2].legend()
axes[2].grid(alpha=0.3)

# Mark actual drift in every panel
for ax in axes:
    ax.axvline(
        DRIFT_POINT,
        color="red",
        linestyle="--",
        label="Actual Drift"
    )
    ax.legend()

plt.tight_layout()

plt.savefig(
    "results/figures/hybrid_signal_comparison.png",
    dpi=300
)

plt.show()

print("Hybrid signal visualization saved successfully!")
