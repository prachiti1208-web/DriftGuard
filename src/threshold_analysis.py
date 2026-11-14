
import pandas as pd

df = pd.read_csv(
    "results/tables/hybrid_detector_results.csv"
)

TRUE_DRIFT_POINT = 1000

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

print("\n========== DRIFTGUARD THRESHOLD ANALYSIS ==========")

for threshold in thresholds:

    alarms = df.loc[
        df["hybrid_score"] >= threshold,
        "record"
    ].tolist()

    false_alarms = [
        point for point in alarms
        if point < TRUE_DRIFT_POINT
    ]

    valid_detections = [
        point for point in alarms
        if point >= TRUE_DRIFT_POINT
    ]

    if valid_detections:
        first_detection = valid_detections[0]
        delay = first_detection - TRUE_DRIFT_POINT
    else:
        first_detection = None
        delay = None

    print(f"\nThreshold: {threshold}")
    print("False alarm count:", len(false_alarms))
    print("First detection:", first_detection)
    print("Detection delay:", delay)

    if false_alarms:
        print("First false alarms:", false_alarms[:10])
