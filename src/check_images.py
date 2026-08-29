from pathlib import Path
from PIL import Image

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

bad_images = []
total_images = 0

for split in ["Train", "Test"]:
    split_path = dataset_path / split

    for class_name in ["Fire", "Smoke", "Neutral"]:
        class_path = split_path / class_name

        for image_path in class_path.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in valid_extensions:
                bad_images.append((image_path, "Unsupported extension"))
                continue

            total_images += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

            except Exception as e:
                bad_images.append((image_path, str(e)))


print("\n================================")
print("IMAGE DATASET CHECK")
print("================================")

print(f"Images checked: {total_images}")
print(f"Bad images found: {len(bad_images)}")

if bad_images:
    print("\nBad images:")

    for path, error in bad_images:
        print(f"\n{path}")
        print(f"Reason: {error}")

else:
    print("\nAll images are valid!")

print("\nCheck completed.")