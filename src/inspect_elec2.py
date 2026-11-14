
from river import datasets
from itertools import islice

# Load the real-world electricity dataset
dataset = datasets.Elec2()

print("\n========== ELEC2 DATASET INFORMATION ==========")

print("Dataset:", dataset)
print("Number of samples:", dataset.n_samples)
print("Number of features:", dataset.n_features)
print("Number of classes:", dataset.n_classes)

print("\n========== FIRST 5 RECORDS ==========")

for i, (features, target) in enumerate(islice(dataset, 5), start=1):

    print(f"\nRecord {i}")
    print("Features:", features)
    print("Target:", target)
