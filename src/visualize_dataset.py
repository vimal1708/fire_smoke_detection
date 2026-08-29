import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# DATASET VISUALIZATION
# ============================================

dataset_path = Path("dataset/FIRE-SMOKE-DATASET")
train_path = dataset_path / "Train"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("\n================================")
print("DATASET VISUALIZATION")
print("================================")

# Load a small batch of images
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")

# Get one batch
images, labels = next(iter(train_ds))

# ============================================
# Display 12 images
# ============================================

plt.figure(figsize=(12, 9))

for i in range(12):

    ax = plt.subplot(3, 4, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i].numpy()])

    plt.axis("off")

plt.suptitle("Fire / Smoke / Neutral Dataset Samples")

plt.tight_layout()

plt.show()

print("\n================================")
print("VISUALIZATION COMPLETE")
print("================================")