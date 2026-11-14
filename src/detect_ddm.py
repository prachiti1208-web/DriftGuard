
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from river.drift.binary import DDM

# -----------------------------------
# 1. Load baseline predictions
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

input_path = ROOT / "results" / "tables" / "baseline_predictions.csv"

data = pd.read_csv(input_path)

# -----------------------------------
# 2. Calculate prediction errors
# -----------------------------------

data["error"] = (
    data["actual"] != data["predicted"]
).astype(int)

# -----------------------------------
# 3. Initialize DDM
# -----------------------------------

detector = DDM(
    warm_start=30,
    warning_threshold=2.0,
    drift_threshold=3.0
)

warnings = []
drift_points = []

# -----------------------------------
# 4. Monitor errors one by one
# -----------------------------------

for _, row in data.iterrows():

    error = int(row["error"])
    record = int(row["record"])

    detector.update(error)

    if detector.warning_detected:
        warnings.append(record)

    if detector.drift_detected:
        drift_points.append(record)

# -----------------------------------
# 5. Display detection results
# -----------------------------------

TRUE_DRIFT_POINT = 1000

print("========== DDM RESULTS ==========")

print("Actual drift point:", TRUE_DRIFT_POINT)

print("Total warnings:", len(warnings))
print("Warning records:", warnings[:20])

print("\nTotal drift detections:", len(drift_points))
print("Detected drift records:", drift_points[:20])


# Separate false alarms and actual drift detections

false_alarms = [
    point for point in drift_points
    if point < TRUE_DRIFT_POINT
]

valid_detections = [
    point for point in drift_points
    if point >= TRUE_DRIFT_POINT
]

print("\n========== DDM EVALUATION ==========")

print("Actual drift point:", TRUE_DRIFT_POINT)

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

# -----------------------------------
# 6. Visualize error rate
# -----------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    data["record"],
    data["rolling_accuracy"],
    label="Rolling Accuracy"
)

plt.axvline(
    x=TRUE_DRIFT_POINT,
    color="red",
    linestyle="--",
    label="Actual Drift"
)

for i, point in enumerate(drift_points):
    plt.axvline(
        x=point,
        color="green",
        linestyle=":",
        label="DDM Detection" if i == 0 else None
    )

plt.xlabel("Record Number")
plt.ylabel("Rolling Accuracy")
plt.title("DDM Drift Detection Results")

plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = ROOT / "results" / "figures" / "ddm_detection.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nGraph saved at:", output_path)
