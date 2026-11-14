
import pandas as pd
from river import preprocessing, linear_model

# Load real-world dataset
df = pd.read_csv("data/processed/elec2_real_world.csv")

# Input features (exclude target and record index)
FEATURES = [
    "date",
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer"
]

INITIAL_TRAIN_SIZE = 1000

# River online learning model
model = (
    preprocessing.StandardScaler()
    | linear_model.LogisticRegression()
)

print("========== ELEC2 ONLINE BASELINE ==========")

# Initial training using first 1000 records
for i in range(INITIAL_TRAIN_SIZE):
    row = df.iloc[i]

    x = {
        feature: float(row[feature])
        for feature in FEATURES
    }

    y = bool(row["target"])

    model.learn_one(x, y)

print(f"Initial training records: {INITIAL_TRAIN_SIZE}")

# Prequential evaluation
results = []

correct = 0
total = 0

for i in range(INITIAL_TRAIN_SIZE, len(df)):

    row = df.iloc[i]

    x = {
        feature: float(row[feature])
        for feature in FEATURES
    }

    actual = bool(row["target"])

    # Step 1: Predict before learning the current label
    probabilities = model.predict_proba_one(x)

    prediction = (
        max(probabilities, key=probabilities.get)
        if probabilities
        else False
    )

    confidence = probabilities.get(prediction, 0.0)

    # Step 2: Evaluate prediction
    is_correct = prediction == actual

    correct += int(is_correct)
    total += 1

    # Step 3: Update model after observing actual label
    model.learn_one(x, actual)

    results.append({
        "record": int(row["record"]),
        "actual": actual,
        "predicted": prediction,
        "prediction_confidence": confidence,
        "correct": int(is_correct)
    })

# Convert predictions to DataFrame
results_df = pd.DataFrame(results)

# Rolling accuracy over 200 predictions
results_df["rolling_accuracy"] = (
    results_df["correct"]
    .rolling(window=200, min_periods=1)
    .mean()
)

# Save predictions
output_path = "results/tables/elec2_baseline_predictions.csv"

results_df.to_csv(output_path, index=False)

# Evaluation summary
accuracy = correct / total

print("\n========== BASELINE RESULTS ==========")
print("Evaluation records:", total)
print(f"Prequential accuracy: {accuracy:.4f}")
print(f"Prequential accuracy (%): {accuracy * 100:.2f}%")
print(
    "Mean prediction confidence:",
    round(results_df["prediction_confidence"].mean(), 4)
)

print("\nFirst 5 predictions:")
print(results_df.head().to_string(index=False))

print("\nLast 5 predictions:")
print(results_df.tail().to_string(index=False))

print("\nSaved predictions to:", output_path)
print("\nBaseline training completed.")
