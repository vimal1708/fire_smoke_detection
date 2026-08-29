from pathlib import Path
from PIL import Image

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

bad_ranges = {
    "Train": range(844, 900),
    "Test": range(0, 100)
}

print("\n================================")
print("TIFF IMAGE VALIDATION")
print("================================")

valid = 0
invalid = 0

for split, image_numbers in bad_ranges.items():

    folder = dataset_path / split / "Neutral"

    for number in image_numbers:

        image_path = folder / f"image_{number}.jpg"

        try:
            with Image.open(image_path) as image:

                print(
                    f"{split}/Neutral/image_{number}.jpg "
                    f"-> format={image.format}, "
                    f"size={image.size}, "
                    f"mode={image.mode}"
                )

                image.verify()

            valid += 1

        except Exception as e:

            invalid += 1

            print(
                f"{split}/Neutral/image_{number}.jpg "
                f"-> INVALID: {e}"
            )

print("\n================================")
print("RESULT")
print("================================")

print(f"Valid images:   {valid}")
print(f"Invalid images: {invalid}")

print("\nCheck completed.")