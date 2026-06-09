"""
check_fear.py  —  Evaluate fear class performance after retraining
===================================================================
Usage:
    python check_fear.py
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# MUST import FocalLoss BEFORE loading model
from focal_loss import FocalLoss

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_TEST  = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\test"
MODEL_PATH = "models/emotion_mobilenet.keras"

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"FocalLoss": FocalLoss}
)
print("✓ Model loaded\n")

# ──────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────
datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = datagen.flow_from_directory(
    DATA_TEST, target_size=(160, 160),
    color_mode="rgb", batch_size=32,
    class_mode="categorical", shuffle=False
)

print("Running predictions...")
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
labels = list(test_gen.class_indices.keys())

# ──────────────────────────────────────────────
# FULL REPORT
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FULL CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=labels))

# ──────────────────────────────────────────────
# FEAR SPECIFIC
# ──────────────────────────────────────────────
print("=" * 55)
print("  FEAR CLASS SPECIFIC ANALYSIS")
print("=" * 55)

fear_idx  = labels.index('fear')
fear_true = (np.array(y_true) == fear_idx)
fear_pred = (np.array(y_pred) == fear_idx)

precision = (fear_true & fear_pred).sum() / fear_pred.sum() if fear_pred.sum() > 0 else 0
recall    = (fear_true & fear_pred).sum() / fear_true.sum() if fear_true.sum() > 0 else 0
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n  Fear test samples:   {fear_true.sum()}")
print(f"  Correctly predicted: {(fear_true & fear_pred).sum()}")
print(f"  Fear Precision:      {precision:.4f}")
print(f"  Fear Recall:         {recall:.4f}")
print(f"  Fear F1-Score:       {f1:.4f}")

print("\n  What fear gets predicted as:")
fear_preds = y_pred[np.where(fear_true)[0]]
for i, label in enumerate(labels):
    count  = (fear_preds == i).sum()
    pct    = count / len(fear_preds) * 100 if len(fear_preds) > 0 else 0
    marker = " ← correct" if label == 'fear' else ""
    print(f"    {label:<12}: {count:>3} ({pct:.1f}%){marker}")

overall_acc = (np.array(y_true) == np.array(y_pred)).mean()
print(f"\n  Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print("=" * 55)