import tensorflow as tf

# -----------------------------
# Dataset paths
# -----------------------------
train_dir = "dataset/FIRE-SMOKE-DATASET/Train"
test_dir = "dataset/FIRE-SMOKE-DATASET/Test"

# -----------------------------
# Image settings
# -----------------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
# -----------------------------
# Load training dataset
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

# -----------------------------
# Load validation dataset
# -----------------------------
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=True
)

# -----------------------------
# Load test dataset
# -----------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Print dataset information
# -----------------------------
print("\nClasses:")
print(train_dataset.class_names)

print("\nDataset information:")
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))
print("Test batches:", len(test_dataset))

# -----------------------------
# Normalize pixel values
# -----------------------------
# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# -----------------------------
# Normalize and augment training data
# -----------------------------
def preprocess_train(images, labels):
    images = images / 255.0
    images = data_augmentation(images, training=True)
    return images, labels


# -----------------------------
# Normalize validation and test data
# -----------------------------
def preprocess_eval(images, labels):
    images = images / 255.0
    return images, labels


train_dataset = train_dataset.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE
)

# -----------------------------
# Improve data pipeline speed
# -----------------------------
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print("\nPreprocessing and augmentation completed.")