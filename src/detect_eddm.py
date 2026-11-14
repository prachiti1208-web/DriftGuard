
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from river.drift.binary import EDDM


# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "tables" / "baseline_predictions.csv"
OUTPUT_FIGURE = ROOT / "results" / "figures" / "eddm_detection.png"

# Known ground-truth drift location
TRUE_DRIFT_POINT = 1000


# Load baseline predictions
df = pd.read_csv(INPUT_FILE)

print("Baseline predictions loaded successfully!")
print("Total prediction records:", len(df))


# Initialize EDDM
detector = EDDM()

warnings = []
drift_points = []


# Process prediction errors sequentially
# Use the original record number when it is available.
if "record" in df.columns:
    record_numbers = df["record"].tolist()
else:
    record_numbers = df.index.tolist()

for position, row in df.iterrows():

    actual = int(row["actual"])
    predicted = int(row["predicted"])

    error = int(actual != predicted)

    record = record_numbers[position]

    detector.update(error)

    if detector.warning_detected:
        warnings.append(record)

    if detector.drift_detected:
        drift_points.append(record)


# Separate false alarms and actual drift detections
false_alarms = [
    point for point in drift_points
    if point < TRUE_DRIFT_POINT
]

valid_detections = [
    point for point in drift_points
    if point >= TRUE_DRIFT_POINT
]


# Display results
print("\n========== EDDM RESULTS ==========")

print("Actual drift point:", TRUE_DRIFT_POINT)

print("Total warnings:", len(warnings))
print("Warning records:", warnings[:20])

print("Total drift detections:", len(drift_points))
print("Detected drift records:", drift_points)

print("\n========== EDDM EVALUATION ==========")

print("False alarms:", false_alarms)
print("Number of false alarms:", len(false_alarms))

print("Valid detections:", valid_detections)

if valid_detections:

    first_detection = valid_detections[0]
    delay = first_detection - TRUE_DRIFT_POINT

    print("First valid detection:", first_detection)
    print("Detection delay:", delay, "records")

else:
    print("Actual drift was not detected.")


# Visualize EDDM detections
plt.figure(figsize=(14, 6))

if "rolling_accuracy" in df.columns:
    plt.plot(
        record_numbers,
        df["rolling_accuracy"],
        label="Rolling Accuracy",
        color="blue"
    )

plt.axvline(
    x=TRUE_DRIFT_POINT,
    color="green",
    linestyle="--",
    label="Actual Drift"
)

for i, point in enumerate(drift_points):
    plt.axvline(
        x=point,
        color="red",
        linestyle=":",
        alpha=0.7,
        label="EDDM Detection" if i == 0 else ""
    )

plt.title("EDDM Concept Drift Detection")
plt.xlabel("Record Number")
plt.ylabel("Rolling Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.show()

print("\nGraph saved at:", OUTPUT_FIGURE)
