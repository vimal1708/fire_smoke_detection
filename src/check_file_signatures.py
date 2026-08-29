from pathlib import Path

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

files_to_check = [
    dataset_path / "Train" / "Fire" / "image_0.jpg",
    dataset_path / "Train" / "Smoke" / "image_0.jpg",
    dataset_path / "Train" / "Neutral" / "image_844.jpg",
    dataset_path / "Test" / "Neutral" / "image_0.jpg",
]

print("\n================================")
print("IMAGE FILE SIGNATURE CHECK")
print("================================")

for image_path in files_to_check:

    print(f"\nFile: {image_path}")

    if not image_path.exists():
        print("Status: FILE NOT FOUND")
        continue

    with open(image_path, "rb") as file:
        header = file.read(16)

    print("First 16 bytes:", header.hex(" ").upper())

    if header.startswith(b"\xFF\xD8\xFF"):
        print("Detected format: JPEG")

    elif header.startswith(b"\x89PNG"):
        print("Detected format: PNG")

    elif header.startswith(b"GIF"):
        print("Detected format: GIF")

    elif header.startswith(b"RIFF"):
        print("Detected format: RIFF/WebP or similar")

    elif header.startswith(b"BM"):
        print("Detected format: BMP")

    else:
        print("Detected format: UNKNOWN")

print("\n================================")
print("Signature check completed.")
print("================================")