from pathlib import Path

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

bad_ranges = {
    "Train": range(844, 900),
    "Test": range(0, 100)
}

print("\n================================")
print("BAD IMAGE FILE INSPECTION")
print("================================")

total = 0

for split, image_numbers in bad_ranges.items():

    folder = dataset_path / split / "Neutral"

    print(f"\n--- {split}/Neutral ---")

    for number in image_numbers:

        image_path = folder / f"image_{number}.jpg"

        if not image_path.exists():
            print(f"Missing: {image_path}")
            continue

        total += 1

        size = image_path.stat().st_size

        print(f"image_{number}.jpg -> {size:,} bytes")

print("\n================================")
print(f"Files inspected: {total}")
print("================================")