import tensorflow as tf
import pickle
from pathlib import Path

from prepare_dataset import prepare_datasets
from build_cnn import build_cnn


# ==========================================
# SETTINGS
# ==========================================

EPOCHS = 30


# ==========================================
# START TRAINING
# ==========================================

print("\n================================")
print("IMPROVED CNN TRAINING")
print("================================")


# ==========================================
# 1. PREPARE DATASET
# ==========================================

train_dataset, validation_dataset, test_dataset, class_names = \
    prepare_datasets()


print("\nClasses:", class_names)


# ==========================================
# 2. BUILD CNN
# ==========================================

model = build_cnn(
    input_shape=(224, 224, 3),
    num_classes=len(class_names)
)


# ==========================================
# 3. MODEL SUMMARY
# ==========================================

print("\n================================")
print("MODEL SUMMARY")
print("================================")

model.summary()


# ==========================================
# 4. CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\n================================")
print("STARTING IMPROVED CNN TRAINING")
print("================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_learning_rate
    ]
)


# ==========================================
# 6. EVALUATE TEST DATA
# ==========================================

print("\n================================")
print("EVALUATING TEST DATA")
print("================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


# ==========================================
# 7. FINAL TEST RESULTS
# ==========================================

print("\n================================")
print("FINAL TEST RESULTS")
print("================================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# ==========================================
# 8. CREATE MODELS DIRECTORY
# ==========================================

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


# ==========================================
# 9. SAVE MODEL
# ==========================================

model_path = models_dir / "improved_fire_smoke_cnn.keras"

model.save(model_path)


print("\n================================")
print("IMPROVED MODEL SAVED")
print("================================")

print(f"Saved to: {model_path}")


# ==========================================
# 10. SAVE TRAINING HISTORY
# ==========================================

history_path = models_dir / "improved_training_history.pkl"

with open(history_path, "wb") as f:
    pickle.dump(history.history, f)


print("\n================================")
print("TRAINING HISTORY SAVED")
print("================================")

print(f"Saved to: {history_path}")


# ==========================================
# 11. BEST VALIDATION RESULT
# ==========================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)


print("\n================================")
print("BEST VALIDATION RESULT")
print("================================")

print(f"Best Epoch               : {best_epoch}")
print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)


# ==========================================
# 12. TRAINING COMPLETE
# ==========================================

print("\n================================")
print("IMPROVED CNN TRAINING COMPLETE")
print("================================")