from pathlib import Path
from PIL import Image

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

bad_ranges = {
    "Train": range(844, 900),
    "Test": range(0, 100)
}

converted = 0
failed = 0

print("\n================================")
print("TIFF TO JPEG CONVERSION")
print("================================")

for split, image_numbers in bad_ranges.items():

    folder = dataset_path / split / "Neutral"

    for number in image_numbers:

        image_path = folder / f"image_{number}.jpg"

        try:
            with Image.open(image_path) as image:

                # Convert to RGB to guarantee JPEG compatibility
                image = image.convert("RGB")

                # Save temporary JPEG
                temp_path = folder / f"image_{number}_temp.jpg"

                image.save(
                    temp_path,
                    format="JPEG",
                    quality=95
                )

            # Replace original TIFF-disguised file
            image_path.unlink()
            temp_path.rename(image_path)

            converted += 1

            print(
                f"Converted: "
                f"{split}/Neutral/image_{number}.jpg"
            )

        except Exception as e:

            failed += 1

            print(
                f"FAILED: "
                f"{split}/Neutral/image_{number}.jpg"
            )
            print(f"Error: {e}")

print("\n================================")
print("CONVERSION RESULT")
print("================================")

print(f"Converted successfully: {converted}")
print(f"Failed:                  {failed}")

print("\nConversion completed.")