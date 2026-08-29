import tensorflow as tf
from pathlib import Path


# ============================================
# DATASET PREPARATION SETTINGS
# ============================================

DATASET_PATH = Path("dataset/FIRE-SMOKE-DATASET")

TRAIN_PATH = DATASET_PATH / "Train"
TEST_PATH = DATASET_PATH / "Test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================
# PREPARE DATASETS FUNCTION
# ============================================

def prepare_datasets():

    print("\n================================")
    print("DATASET PREPARATION")
    print("================================")

    # ----------------------------------------
    # Training dataset
    # ----------------------------------------

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training"
    )

    # ----------------------------------------
    # Validation dataset
    # ----------------------------------------

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="validation"
    )

    # ----------------------------------------
    # Test dataset
    # ----------------------------------------

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----------------------------------------
    # Class names
    # ----------------------------------------

    class_names = train_ds.class_names

    print("\nClass names:")
    print(class_names)

    # ----------------------------------------
    # Dataset information
    # ----------------------------------------

    print("\n================================")
    print("DATASET INFORMATION")
    print("================================")

    print(f"Image size        : {IMAGE_SIZE}")
    print(f"Batch size        : {BATCH_SIZE}")

    print(
        f"Training batches  : "
        f"{tf.data.experimental.cardinality(train_ds).numpy()}"
    )

    print(
        f"Validation batches: "
        f"{tf.data.experimental.cardinality(val_ds).numpy()}"
    )

    print(
        f"Test batches      : "
        f"{tf.data.experimental.cardinality(test_ds).numpy()}"
    )

    # ----------------------------------------
    # Check one batch
    # ----------------------------------------

    images, labels = next(iter(train_ds))

    print("\n================================")
    print("BATCH CHECK")
    print("================================")

    print(f"Image batch shape : {images.shape}")
    print(f"Label batch shape : {labels.shape}")
    print(f"Image data type   : {images.dtype}")
    print(f"Label data type   : {labels.dtype}")

    print("\nSample labels:")
    print(labels[:10].numpy())

    print("\n================================")
    print("PREPARATION COMPLETE")
    print("================================")

    return train_ds, val_ds, test_ds, class_names


# ============================================
# TEST DATASET PREPARATION
# ============================================

if __name__ == "__main__":

    prepare_datasets()