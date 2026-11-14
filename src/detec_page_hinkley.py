
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from river.drift import PageHinkley


# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "results" / "tables" / "baseline_predictions.csv"
OUTPUT_FIGURE = ROOT / "results" / "figures" / "page_hinkley_detection.png"

# Actual drift location
TRUE_DRIFT_POINT = 1000


# Load baseline predictions
df = pd.read_csv(INPUT_FILE)

print("Baseline predictions loaded successfully!")
print("Total prediction records:", len(df))


# Initialize Page-Hinkley detector
detector = PageHinkley()

drift_points = []


# Feed prediction errors sequentially
for _, row in df.iterrows():

    actual = int(row["actual"])
    predicted = int(row["predicted"])

    error = int(actual != predicted)

    record = int(row["record"])

    detector.update(error)

    if detector.drift_detected:
        drift_points.append(record)


# Separate false alarms and valid detections
false_alarms = [
    point for point in drift_points
    if point < TRUE_DRIFT_POINT
]

valid_detections = [
    point for point in drift_points
    if point >= TRUE_DRIFT_POINT
]


# Display results
print("\n========== PAGE-HINKLEY RESULTS ==========")

print("Actual drift point:", TRUE_DRIFT_POINT)

print("Total drift detections:", len(drift_points))
print("Detected drift records:", drift_points)

print("\n========== PAGE-HINKLEY EVALUATION ==========")

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


# Visualize Page-Hinkley detections
plt.figure(figsize=(14, 6))

plt.plot(
    df["record"],
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
        label="Page-Hinkley Detection" if i == 0 else ""
    )

plt.title("Page-Hinkley Concept Drift Detection")
plt.xlabel("Record Number")
plt.ylabel("Rolling Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.show()

print("\nGraph saved at:", OUTPUT_FIGURE)
