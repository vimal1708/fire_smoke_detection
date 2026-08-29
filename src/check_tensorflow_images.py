import tensorflow as tf
from pathlib import Path

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")

bad_images = []
total = 0

for split in ["Train", "Test"]:
    for class_name in ["Fire", "Smoke", "Neutral"]:

        folder = dataset_path / split / class_name

        for image_path in folder.iterdir():

            if not image_path.is_file():
                continue

            total += 1

            try:
                image_bytes = tf.io.read_file(str(image_path))
                image = tf.io.decode_image(
                    image_bytes,
                    channels=3,
                    expand_animations=False
                )

                # Force TensorFlow to actually evaluate the image
                _ = image.numpy()

            except Exception as e:
                bad_images.append((image_path, str(e)))


print("\n================================")
print("TENSORFLOW IMAGE CHECK")
print("================================")

print(f"Images checked: {total}")
print(f"Bad images found: {len(bad_images)}")

if bad_images:

    print("\nProblematic images:")

    for path, error in bad_images:
        print("\nFile:", path)
        print("Error:", error)

else:
    print("\nAll images can be decoded by TensorFlow!")

print("\nCheck completed.")