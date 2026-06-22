"""Recompute Model 3's confusion structure for an evidence-grounded discussion."""
import os, json, random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG, BS = (150, 150), 32
test = tf.keras.utils.image_dataset_from_directory(
    "data/seg_test/seg_test", image_size=IMG, batch_size=BS, shuffle=False)
CLASS_NAMES = test.class_names
norm = tf.keras.layers.Rescaling(1/255)
test_n = test.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)

y_true = np.concatenate([y.numpy() for _, y in test_n])
detail = {"class_names": CLASS_NAMES, "models": {}}

for name, path in [("Model 1", "model1.keras"), ("Model 2", "model2.keras"),
                   ("Model 3", "model3.keras"), ("Model 4", "model4_mobilenet.keras")]:
    model = tf.keras.models.load_model(path)
    y_pred = np.argmax(model.predict(test_n, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # top off-diagonal confusions
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])
    recall = {CLASS_NAMES[i]: round(cm[i, i] / cm[i].sum(), 3) for i in range(len(CLASS_NAMES))}
    detail["models"][name] = {"test_acc": round(float(acc), 4),
                              "top_confusions": pairs[:5], "per_class_recall": recall}
    print(f"\n{name}  (test acc {acc:.4f})")
    print("  worst class recall:", min(recall, key=recall.get), recall[min(recall, key=recall.get)])
    print("  top confusions (true -> pred, n):")
    for a, b, n in pairs[:5]:
        print(f"     {a:9s} -> {b:9s}  {n}")

with open("confusion_detail.json", "w") as f:
    json.dump(detail, f, indent=2)
print("\nSaved confusion_detail.json")
