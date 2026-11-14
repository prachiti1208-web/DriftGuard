
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------
# 1. Command-line configuration
# -----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    required=True
)

parser.add_argument(
    "--output",
    type=str,
    required=True
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()

ROOT = Path(__file__).resolve().parent.parent

data_path = ROOT / args.input
output_path = ROOT / args.output

# -----------------------------------
# 2. Load dataset
# -----------------------------------

data = pd.read_csv(data_path)

X = data[["feature_1", "feature_2"]]
y = data["target"]

INITIAL_TRAIN_SIZE = 200
WINDOW_SIZE = 100

if len(data) <= INITIAL_TRAIN_SIZE:
    raise ValueError("Dataset is too small for initial training.")

if "drift" not in data.columns:
    raise ValueError("Dataset must contain the ground-truth drift column.")

drift_records = data.index[data["drift"] == 1]

if len(drift_records) == 0:
    raise ValueError("No drift point found.")

DRIFT_POINT = int(drift_records[0])

# -----------------------------------
# 3. Create model
# -----------------------------------

model = SGDClassifier(
    loss="log_loss",
    random_state=args.seed,
    learning_rate="constant",
    eta0=0.01
)

# -----------------------------------
# 4. Initial training
# -----------------------------------

model.partial_fit(
    X.iloc[:INITIAL_TRAIN_SIZE],
    y.iloc[:INITIAL_TRAIN_SIZE],
    classes=np.array([0, 1])
)

print("Initial model training completed!")
print("Random seed:", args.seed)
print("True drift point:", DRIFT_POINT)

# -----------------------------------
# 5. Prequential evaluation
# -----------------------------------

results = []

actual_values = []
predicted_values = []

for i in range(INITIAL_TRAIN_SIZE, len(data)):

    x_current = X.iloc[[i]]
    y_current = y.iloc[i]

    # Predict before observing the current label
    prediction = model.predict(x_current)[0]

    probabilities = model.predict_proba(x_current)[0]

    class_to_index = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    probability_0 = probabilities[class_to_index[0]]
    probability_1 = probabilities[class_to_index[1]]

    confidence = probabilities[
        class_to_index[prediction]
    ]

    # Record prediction
    actual_values.append(y_current)
    predicted_values.append(prediction)

    # Rolling accuracy
    start = max(0, len(actual_values) - WINDOW_SIZE)

    rolling_accuracy = accuracy_score(
        actual_values[start:],
        predicted_values[start:]
    )

    results.append({
        "record": i,
        "feature_1": x_current["feature_1"].iloc[0],
        "feature_2": x_current["feature_2"].iloc[0],
        "actual": y_current,
        "predicted": prediction,
        "probability_class_0": probability_0,
        "probability_class_1": probability_1,
        "prediction_confidence": confidence,
        "rolling_accuracy": rolling_accuracy
    })

    # Update only after recording the prediction
    model.partial_fit(x_current, [y_current])

results = pd.DataFrame(results)

# -----------------------------------
# 6. Performance evaluation
# -----------------------------------

before = results[results["record"] < DRIFT_POINT]
after = results[results["record"] >= DRIFT_POINT]

print("\n========== MODEL PERFORMANCE ==========")

print(
    "Accuracy before drift:",
    round(accuracy_score(before["actual"], before["predicted"]), 4)
)

print(
    "Accuracy after drift:",
    round(accuracy_score(after["actual"], after["predicted"]), 4)
)

print(
    "Average confidence:",
    round(results["prediction_confidence"].mean(), 4)
)

# -----------------------------------
# 7. Save predictions
# -----------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)

print("\nPredictions saved successfully!")
print("Output:", output_path)
print("Total prediction records:", len(results))
