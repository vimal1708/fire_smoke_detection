from pathlib import Path

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

train_path = dataset_path / "Train"
test_path = dataset_path / "Test"

classes = ["Fire", "Smoke", "Neutral"]

print("Checking dataset...\n")

for split_name, split_path in [("TRAIN", train_path), ("TEST", test_path)]:
    print(f"{split_name} DATASET")

    total = 0

    for class_name in classes:
        class_path = split_path / class_name

        images = [
            file for file in class_path.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{class_name}: {len(images)} images")
        total += len(images)

    print(f"Total: {total}\n")

print("Dataset check completed!")