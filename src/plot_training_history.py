import pickle
from pathlib import Path

import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

HISTORY_PATH = Path("models/training_history.pkl")


print("\n================================")
print("TRAINING HISTORY ANALYSIS")
print("================================")


# ==========================================
# CHECK HISTORY FILE
# ==========================================

if not HISTORY_PATH.exists():
    print("\nTraining history file not found!")
    print(f"Expected: {HISTORY_PATH}")
    print("\nRun train.py first.")
    exit()


# ==========================================
# LOAD HISTORY
# ==========================================

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)


# ==========================================
# GET METRICS
# ==========================================

train_accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

train_loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_accuracy) + 1)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print(f"\nEpochs completed: {len(train_accuracy)}")

print("\nFinal Training Accuracy : "
      f"{train_accuracy[-1]:.4f}")

print("Final Validation Accuracy: "
      f"{val_accuracy[-1]:.4f}")

print("\nFinal Training Loss      : "
      f"{train_loss[-1]:.4f}")

print("Final Validation Loss   : "
      f"{val_loss[-1]:.4f}")


# ==========================================
# BEST VALIDATION ACCURACY
# ==========================================

best_epoch = val_accuracy.index(max(val_accuracy)) + 1
best_val_accuracy = max(val_accuracy)

print("\n================================")
print("BEST VALIDATION RESULT")
print("================================")

print(f"Best Epoch             : {best_epoch}")
print(f"Best Validation Accuracy: "
      f"{best_val_accuracy:.4f}")


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/training_validation_accuracy.png",
    dpi=300
)

plt.show()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/training_validation_loss.png",
    dpi=300
)

plt.show()


# ==========================================
# COMPLETE
# ==========================================

print("\n================================")
print("HISTORY ANALYSIS COMPLETE")
print("================================")

print("Accuracy graph saved to:")
print("models/training_validation_accuracy.png")

print("\nLoss graph saved to:")
print("models/training_validation_loss.png")