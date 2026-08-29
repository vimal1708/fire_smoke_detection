import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# CLASS DISTRIBUTION
# ============================================

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

classes = ["Fire", "Smoke", "Neutral"]

train_counts = []
test_counts = []

for class_name in classes:

    train_folder = dataset_path / "Train" / class_name
    test_folder = dataset_path / "Test" / class_name

    train_count = sum(1 for file in train_folder.iterdir() if file.is_file())
    test_count = sum(1 for file in test_folder.iterdir() if file.is_file())

    train_counts.append(train_count)
    test_counts.append(test_count)

# ============================================
# Print results
# ============================================

print("\n================================")
print("CLASS DISTRIBUTION")
print("================================")

for i, class_name in enumerate(classes):

    print(
        f"{class_name:<8} : "
        f"Train = {train_counts[i]}, "
        f"Test = {test_counts[i]}"
    )

# ============================================
# Plot
# ============================================

x = range(len(classes))

plt.figure(figsize=(8, 5))

plt.bar(
    [i - 0.2 for i in x],
    train_counts,
    width=0.4,
    label="Train"
)

plt.bar(
    [i + 0.2 for i in x],
    test_counts,
    width=0.4,
    label="Test"
)

plt.xticks(list(x), classes)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.title("Fire-SMoke-Neutral Dataset Distribution")

plt.legend()

plt.tight_layout()

plt.show()

print("\n================================")
print("DISTRIBUTION CHECK COMPLETE")
print("================================")