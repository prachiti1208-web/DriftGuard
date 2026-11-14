
import subprocess
import sys
from pathlib import Path

# -----------------------------------
# Project configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

SEEDS = range(42, 52)

DATA_DIR = ROOT / "data" / "processed" / "streams"
RESULT_DIR = ROOT / "results" / "tables" / "streams"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# -----------------------------------
# Run experiments
# -----------------------------------

for seed in SEEDS:

    print("\n" + "=" * 50)
    print(f"RUNNING EXPERIMENT FOR SEED {seed}")
    print("=" * 50)

    dataset = DATA_DIR / f"abrupt_seed_{seed}.csv"

    predictions = RESULT_DIR / f"baseline_seed_{seed}.csv"

    # Generate dataset
    generate_command = [
        PYTHON,
        str(ROOT / "src" / "generate_stream.py"),
        "--seed", str(seed),
        "--output",
        str(dataset.relative_to(ROOT))
    ]

    subprocess.run(
        generate_command,
        cwd=ROOT,
        check=True
    )

    # Train baseline model
    train_command = [
        PYTHON,
        str(ROOT / "src" / "train_baseline.py"),
        "--input",
        str(dataset.relative_to(ROOT)),
        "--output",
        str(predictions.relative_to(ROOT)),
        "--seed", str(seed)
    ]

    subprocess.run(
        train_command,
        cwd=ROOT,
        check=True
    )

    print(f"Seed {seed} completed successfully!")

print("\n" + "=" * 50)
print("ALL MULTI-SEED EXPERIMENTS COMPLETED!")
print("Seeds:", list(SEEDS))
print("Datasets:", DATA_DIR)
print("Predictions:", RESULT_DIR)
print("=" * 50)
