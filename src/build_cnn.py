import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================
# IMPROVED CNN MODEL SETTINGS
# ============================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3


# ============================================
# BUILD IMPROVED CNN FUNCTION
# ============================================

def build_cnn(
    input_shape=(224, 224, 3),
    num_classes=3
):

    model = models.Sequential([

        # ====================================
        # INPUT
        # ====================================

        layers.Input(shape=input_shape),

        # ====================================
        # DATA AUGMENTATION
        # ====================================

        layers.RandomFlip("horizontal"),

        layers.RandomRotation(0.1),

        layers.RandomZoom(0.1),

        # ====================================
        # CONVOLUTION BLOCK 1
        # ====================================

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            use_bias=False
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D((2, 2)),

        # ====================================
        # CONVOLUTION BLOCK 2
        # ====================================

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            use_bias=False
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D((2, 2)),

        # ====================================
        # CONVOLUTION BLOCK 3
        # ====================================

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            use_bias=False
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D((2, 2)),

        # ====================================
        # CONVOLUTION BLOCK 4
        # ====================================

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            use_bias=False
        ),

        layers.BatchNormalization(),

        layers.Activation("relu"),

        layers.MaxPooling2D((2, 2)),

        # ====================================
        # GLOBAL AVERAGE POOLING
        # ====================================

        layers.GlobalAveragePooling2D(),

        # ====================================
        # FULLY CONNECTED LAYER
        # ====================================

        layers.Dense(
            128,
            activation="relu"
        ),

        # ====================================
        # DROPOUT
        # ====================================

        layers.Dropout(0.5),

        # ====================================
        # OUTPUT
        # ====================================

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])


    # ========================================
    # COMPILE MODEL
    # ========================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )


    return model


# ============================================
# TEST MODEL BUILD
# ============================================

if __name__ == "__main__":

    print("\n================================")
    print("BUILDING IMPROVED CNN MODEL")
    print("================================")

    model = build_cnn(
        input_shape=(224, 224, 3),
        num_classes=3
    )

    print("\n================================")
    print("MODEL SUMMARY")
    print("================================")

    model.summary()

    print("\n================================")
    print("MODEL CONFIGURATION")
    print("================================")

    print("Input size      :", IMAGE_SIZE)
    print("Number of class :", NUM_CLASSES)
    print("Output classes  :", ["Fire", "Neutral", "Smoke"])
    print("Optimizer       : Adam")
    print("Learning rate   : 0.001")
    print("Loss function   : Sparse Categorical Crossentropy")
    print("Regularization  : BatchNormalization + Dropout")
    print("Pooling         : GlobalAveragePooling2D")
    print("Augmentation    : Flip + Rotation + Zoom")

    print("\n================================")
    print("IMPROVED CNN BUILD COMPLETE")
    print("================================")