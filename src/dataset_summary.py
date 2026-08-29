from pathlib import Path

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

classes = ["Fire", "Smoke", "Neutral"]
splits = ["Train", "Test"]

print("\n================================")
print("DATASET SUMMARY")
print("================================")

grand_total = 0

for split in splits:

    print(f"\n--- {split} ---")

    split_total = 0

    for class_name in classes:

        folder = dataset_path / split / class_name

        count = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
        )

        print(f"{class_name:8} : {count}")

        split_total += count

    print(f"Total    : {split_total}")

    grand_total += split_total

print("\n================================")
print(f"Grand Total: {grand_total}")
print("================================")