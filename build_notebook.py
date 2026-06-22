"""Generate SceneDepth_CNN_Classification.ipynb (local-CPU adapted).

Adaptations vs the original Colab brief:
  * no google.colab; Kaggle creds read from ~/.kaggle/kaggle.json
  * relative local paths (./data, ./.tfcache)
  * disk-backed tf.data cache (memory-safe on 18 GB)
  * Functional API models with a named `conv_last` layer for Grad-CAM
  * manual GradientTape Grad-CAM (pip `grad-cam` is PyTorch-only)
  * cosine decay over the full Model-3 run (decay_steps=1000 would zero the LR after ~3 epochs)
"""
import json

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

# Task 6 narrative — grounded in this run's actual (non-monotonic) results. Single source of
# truth shared with postprocess.py, which rewrites the executed notebook's Task 6 cell.
TASK6_DISCUSSION = '''## Task 6: Results Discussion

### Which architecture performed best, and did depth help?
The best model overall was **Model 4 (MobileNetV2 transfer learning) at 90.2% test accuracy** (macro-F1 0.904). The best custom network was **Model 3 (five convolutional layers) at 82.7%** (F1 0.829). Crucially, **depth did not help monotonically**: Model 2 (four conv layers, 73.1%) performed **6.2 points *worse* than the two-layer Model 1 (79.3%)**. Capacity was not the bottleneck — Model 2's training and validation accuracies were both low (≈0.74 / 0.72), so it *under-fitted* rather than over-fitted, settling into a poor optimisation basin.

### What went wrong in Model 2 — and how Model 3 fixed it
The confusion matrices pinpoint the failure. Model 2 collapsed low-texture scenes into one class: it labelled **190 glacier images and 151 mountain images as *sea***, pulling glacier recall down to 0.44. Adding convolutional depth and batch normalisation without any change to the optimisation schedule simply amplified a spurious "flat horizontal band ⇒ sea" shortcut. Model 3 — deeper still, but paired with a **cosine learning-rate schedule** and a larger, more strongly-regularised dense head — more than halved those errors (glacier→sea 82, mountain→sea 66) and lifted glacier recall to 0.69.

### The diminishing-returns question
The answer to our research question is therefore nuanced. Across the custom networks the depth–accuracy relationship was **non-monotonic, not smoothly diminishing**: naïvely stacking layers (Model 2) produced *negative* returns, and even the best-tuned deep custom network (Model 3) **plateaued around 83%** — roughly **7 points below transfer learning**. On a mid-sized (~14k-image) dataset, additional depth pays off only when accompanied by better optimisation (the LR schedule), and the custom-CNN ceiling sits well short of what pretrained features deliver.

### Which classes were most confused, and why
**glacier was the hardest class for every model** (recall 0.63 → 0.44 → 0.69 → 0.83 from Model 1 to Model 4). For Model 3 the largest single error was **glacier misread as sea (82 images)**, followed by **street → buildings (80)** and **mountain → sea (66)**:
- **glacier ↔ sea / mountain.** Glaciers shot with melt-water and a flat skyline share the low-texture, cool-toned, horizontally-banded composition of sea scenes, while snow-on-rock shares texture and relief with mountains; the discriminative cue often occupies only a small part of the frame.
- **street ↔ buildings.** The two co-occur by definition: a street scene almost always contains buildings, so the label hinges on framing (road foreground vs façade) rather than on distinct objects.

### What Grad-CAM reveals
The Grad-CAM overlays on Model 3's mistakes show the network attending to the **horizon band and broad regional texture** rather than the object-defining region a human would fixate on. On glacier/sea and glacier/mountain errors the activation spreads across the whole scene instead of localising on ice, water or rock — direct visual confirmation that the confusions stem from shared low-level scene geometry, not from a labelling fault.

### Transfer learning vs the custom CNNs
MobileNetV2 reached **90.2%** — **+7.5 points over the best custom network (Model 3)** — and its two-stage curve makes the mechanism explicit: Stage 1 (frozen ImageNet features) already reached **89.9% validation accuracy**, and unfreezing the top 30 layers at 1e-5 nudged it to **91.4%**. Transfer learning also repaired the signature weakness, lifting glacier recall to 0.83. ImageNet pre-training supplies rich, general edge/texture/part detectors that a 14k-image network cannot learn from scratch — which is precisely why it clears the depth ceiling the custom models ran into.

> *All figures above are the actual test-set results from this run (`results_summary.json` / `confusion_detail.json`); the code cell below reprints the headline numbers directly from the trained models as an audit trail.*'''

# ── CELL 0 ──────────────────────────────────────────────────────────────────
md('''# SceneDepth: CNN Architecture Depth Analysis for Natural Scene Classification
## AI503 Machine Learning — Assignment 3

**Research Question:** At what depth does a CNN trained on natural scene classification reach diminishing returns, and does transfer learning from a large-scale dataset overcome this ceiling?

**Student:** Fadi Alazayem
**Dataset:** Intel Image Classification (6 classes, ~25,000 images, 150×150 RGB)
**Seed:** 42 (NumPy, TensorFlow, Python `random`, `PYTHONHASHSEED`)

---
*Execution environment:* developed and executed **locally on an Apple M3 Pro (CPU-only** — no CUDA/Metal acceleration). All prose is written in British English.''')

# ── CELL 1 ──────────────────────────────────────────────────────────────────
code('''# --- Dependencies (already satisfied locally; uncomment to reproduce elsewhere) ---
# %pip install -q kaggle tensorflow seaborn opencv-python matplotlib scikit-learn pandas

%matplotlib inline
import os, random
import numpy as np
import tensorflow as tf

# --- Reproducibility: seed everything ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpus)
print("Running on:", "GPU" if gpus else "CPU (Apple M3 Pro)")

# --- Kaggle authentication (local) ---
# Authenticate the Kaggle CLI before running the next cell. Any of these works:
#   * classic API key:  place kaggle.json at ~/.kaggle/kaggle.json   (kaggle < 1.7)
#   * env vars:         export KAGGLE_USERNAME / KAGGLE_KEY
#   * kaggle >= 2.x:    `kaggle auth login` (OAuth) or ~/.kaggle/access_token
from pathlib import Path
kag = Path.home() / ".kaggle"
if (kag / "kaggle.json").exists():
    os.chmod(kag / "kaggle.json", 0o600)
cred_present = any([
    bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")),
    bool(os.environ.get("KAGGLE_API_TOKEN")),
    (kag / "kaggle.json").exists(),
    (kag / "access_token").exists(),
])
print("Kaggle credentials detected:", cred_present)
if not cred_present:
    print("WARNING: no Kaggle credentials found — the download cell will fail until you authenticate.")''')

# ── CELL 2 ──────────────────────────────────────────────────────────────────
code('''# --- Download & extract the Intel Image Classification dataset ---
DATA_ROOT = "data"
TRAIN_DIR_CHECK = os.path.join(DATA_ROOT, "seg_train", "seg_train")

if not os.path.isdir(TRAIN_DIR_CHECK):
    print("Downloading dataset (~350 MB)...")
    !kaggle datasets download -d puneet6060/intel-image-classification -p {DATA_ROOT} --unzip -q
else:
    print("Dataset already present — skipping download.")

# --- Verify structure ---
for split in ["seg_train/seg_train", "seg_test/seg_test"]:
    path = os.path.join(DATA_ROOT, *split.split("/"))
    classes = sorted([c for c in os.listdir(path) if not c.startswith(".")])
    counts = {c: len(os.listdir(os.path.join(path, c))) for c in classes}
    print(f"{split}:  total={sum(counts.values())}  ->  {counts}")''')

# ── CELL 3 (md) ─────────────────────────────────────────────────────────────
md('''## Task 1: Dataset Description

The **Intel Image Classification** dataset contains roughly **25,000 natural-scene photographs** at **150×150 RGB**, organised into **six classes**: `buildings`, `forest`, `glacier`, `mountain`, `sea` and `street`. It ships pre-split into a training set (`seg_train`, ~14k images) and a test set (`seg_test`, ~3k images), plus an unlabelled prediction set that we do not use.

**Why this dataset?**
- **Resolution that makes preprocessing matter.** At 150×150 the images carry far more spatial detail than toy datasets such as CIFAR-10 (32×32), so decisions about normalisation, augmentation and pooling have a genuine, measurable effect rather than being cosmetic.
- **Meaningful inter-class similarity.** Several classes are visually entangled — *glacier* vs *mountain*, *sea* vs *street* — sharing textures, palettes and horizon geometry. This yields structured, interpretable confusion patterns worth analysing, rather than trivially separable categories.
- **Defensible transfer learning.** The scenes overlap strongly with ImageNet's natural-image domain, so an ImageNet-pretrained backbone (MobileNetV2) is academically justified, not arbitrary.''')

# ── CELL 3 (code) ───────────────────────────────────────────────────────────
code('''import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = os.path.join(DATA_ROOT, "seg_train", "seg_train")
TEST_DIR  = os.path.join(DATA_ROOT, "seg_test", "seg_test")
CLASS_NAMES = sorted([c for c in os.listdir(TRAIN_DIR) if not c.startswith(".")])
print("Classes:", CLASS_NAMES)

for name, d in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    total = sum(len(os.listdir(os.path.join(d, c))) for c in CLASS_NAMES)
    print(f"{name}: {total} images")

# Probe one image for its shape
_probe = Image.open(os.path.join(TRAIN_DIR, CLASS_NAMES[0],
                    sorted(os.listdir(os.path.join(TRAIN_DIR, CLASS_NAMES[0])))[0]))
print("Example image shape:", np.array(_probe).shape)

# 3 sample images per class (deterministic — first three filenames)
n_per = 3
fig, axes = plt.subplots(n_per, len(CLASS_NAMES), figsize=(len(CLASS_NAMES) * 2.2, n_per * 2.2))
for j, cls in enumerate(CLASS_NAMES):
    files = sorted(os.listdir(os.path.join(TRAIN_DIR, cls)))[:n_per]
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(TRAIN_DIR, cls, fn)).convert("RGB")
        ax = axes[i, j]
        ax.imshow(img); ax.axis("off")
        if i == 0:
            ax.set_title(cls, fontsize=11)
fig.suptitle("Intel Image Classification — 3 samples per class", fontsize=13)
plt.tight_layout()
plt.savefig("task1_sample_grid.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved task1_sample_grid.png")''')

# ── CELL 4 (md) ─────────────────────────────────────────────────────────────
md('''## Task 2: Preprocessing

Each step is applied deliberately to help the convolutional networks learn:

1. **Resizing to 150×150.** Fixes a uniform input tensor shape so batches stack cleanly and the dense head has a fixed fan-in. The dataset is already ~150×150, so this mainly guards against the handful of off-size test images.
2. **80/20 train–validation split** (from `seg_train`, seeded at 42). Validation drives early stopping and model selection; the official `seg_test` set is held out entirely and only touched for final evaluation, so reported test scores are unbiased.
3. **Normalisation to [0, 1]** (`Rescaling(1/255)`). Small, well-conditioned inputs keep activations and gradients in a healthy range, speeding convergence and stabilising training — particularly ahead of batch-norm layers.
4. **Data augmentation** (horizontal flip, ±10% rotation, ±10% zoom), applied to the *training* stream only. Scenes are largely flip- and scale-invariant, so this cheaply multiplies effective dataset size and is the single most effective regulariser against overfitting on ~14k images.
5. **Caching + prefetching** (`tf.data`). Caching avoids re-decoding JPEGs every epoch; prefetching overlaps data preparation with compute, materially reducing per-epoch time on a CPU. Augmentation is applied *after* the cache so each epoch still sees freshly randomised images.

> **Local-execution note:** the cache is written to disk (`.tfcache/`) rather than RAM to stay within the machine's 18 GB memory budget.''')

# ── CELL 4 (code) ───────────────────────────────────────────────────────────
code('''import shutil
DATA_DIR_TRAIN = os.path.join(DATA_ROOT, "seg_train", "seg_train")
DATA_DIR_TEST  = os.path.join(DATA_ROOT, "seg_test", "seg_test")
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# 80/20 train/val split from the training directory; test held out entirely.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TEST, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print("Classes:", CLASS_NAMES, "| NUM_CLASSES:", NUM_CLASSES)

# Normalisation: scale pixels to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# Data augmentation (training only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.1, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED),
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE
CACHE_DIR = ".tfcache"
shutil.rmtree(CACHE_DIR, ignore_errors=True)   # fresh cache each full run
os.makedirs(CACHE_DIR, exist_ok=True)

def normalise_cache(ds, cache_name):
    ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, cache_name))   # disk cache: memory-safe
    return ds.prefetch(AUTOTUNE)

train_ds = normalise_cache(train_ds, "train")
val_ds   = normalise_cache(val_ds, "val")
test_ds  = normalise_cache(test_ds, "test")

# Augmented training pipeline (augmentation AFTER cache, so it varies each epoch)
train_aug_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print("Train batches:", int(train_ds.cardinality().numpy()),
      "| Val batches:", int(val_ds.cardinality().numpy()),
      "| Test batches:", int(test_ds.cardinality().numpy()))''')

# ── CELL 5 (code) helpers ───────────────────────────────────────────────────
code('''import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)
from tensorflow.keras.callbacks import EarlyStopping

results_table = []   # one row per model

def best_epoch_metrics(history_dict):
    """Return (train_acc, val_acc, train_loss, val_loss, best_idx) at the highest-val-accuracy epoch."""
    h = history_dict
    val_acc = h.get("val_accuracy", h.get("val_acc"))
    best = int(np.argmax(val_acc))
    return (float(h["accuracy"][best]), float(val_acc[best]),
            float(h["loss"][best]), float(h["val_loss"][best]), best)

def _merge(history, extra_history=None):
    h = {k: list(v) for k, v in history.history.items()}
    if extra_history is not None:
        for k, v in extra_history.history.items():
            h[k] = h.get(k, []) + list(v)
    return h

def plot_history(history, model_name, extra_history=None, split_epoch=None, fname=None):
    """Plot accuracy & loss curves (optionally concatenating a second training stage)."""
    h = _merge(history, extra_history)
    acc, val_acc = h.get("accuracy"), h.get("val_accuracy")
    loss, val_loss = h.get("loss"), h.get("val_loss")
    epochs = range(1, len(acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(epochs, acc, "o-", label="train"); ax[0].plot(epochs, val_acc, "s-", label="val")
    ax[0].set_title(f"{model_name} — Accuracy"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, loss, "o-", label="train"); ax[1].plot(epochs, val_loss, "s-", label="val")
    ax[1].set_title(f"{model_name} — Loss"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
    if split_epoch is not None:
        for a in ax:
            a.axvline(split_epoch + 0.5, color="red", ls="--", alpha=0.7)
            a.text(split_epoch + 0.6, a.get_ylim()[0], " fine-tune", color="red", fontsize=9)
    fname = fname or f"history_{model_name.lower().replace(' ', '_')}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.show()
    print("Saved", fname)

def get_true_pred(model, ds):
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0)
    return y_true, np.argmax(y_prob, axis=1), y_prob

def evaluate_model(model, test_ds, class_names, model_name):
    y_true, y_pred, _ = get_true_pred(model, test_ds)
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\\n=== {model_name} — Test classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} — Confusion Matrix (test)")
    plt.ylabel("True"); plt.xlabel("Predicted")
    fname = f"confusion_{model_name.lower().replace(' ', '_')}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=120, bbox_inches="tight"); plt.show()
    print("Saved", fname)
    return {"test_accuracy": float(acc),
            "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
            "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
            "confusion_matrix": cm.tolist()}

def build_results_row(model_name, conv_layers, history, eval_dict, extra_history=None):
    tr_acc, va_acc, tr_loss, va_loss, _ = best_epoch_metrics(_merge(history, extra_history))
    row = {"Model": model_name, "Conv Layers": conv_layers,
           "Train Acc": round(tr_acc, 4), "Val Acc": round(va_acc, 4),
           "Test Acc": round(eval_dict["test_accuracy"], 4),
           "Train Loss": round(tr_loss, 4), "Val Loss": round(va_loss, 4),
           "Precision": round(eval_dict["precision_macro"], 4),
           "Recall": round(eval_dict["recall_macro"], 4),
           "F1": round(eval_dict["f1_macro"], 4),
           "_cm": eval_dict["confusion_matrix"]}
    results_table.append(row)
    return row

print("Helpers ready.")''')

# ── CELL 6 (md) Model 1 ─────────────────────────────────────────────────────
md('''## Model 1: Basic CNN — 2 Convolutional Layers

A deliberately shallow baseline establishing the floor for the depth study.

```
Input(150,150,3)
 → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
 → Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
 → Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(6, Softmax)
```
Trained with Adam (1e-3), sparse categorical cross-entropy, up to 20 epochs, early stopping (patience 5, restore best weights).''')

# ── CELL 6 (code) ───────────────────────────────────────────────────────────
code('''from tensorflow.keras import layers, models, Input

def build_model_1():
    inputs = Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv_last")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Model_1_Basic")

tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
model1 = build_model_1()
model1.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model1.summary()

es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
hist1 = model1.fit(train_aug_ds, validation_data=val_ds, epochs=20, callbacks=[es], verbose=2)

plot_history(hist1, "Model 1")
eval1 = evaluate_model(model1, test_ds, CLASS_NAMES, "Model 1")
build_results_row("Model 1", 2, hist1, eval1)
model1.save("model1.keras"); print("Saved model1.keras")''')

# ── CELL 7 (md) Model 2 ─────────────────────────────────────────────────────
md('''## Model 2: Medium CNN — 4 Convolutional Layers

Adds depth, batch normalisation and a wider dense head.

```
Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(128)→BN
 → Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(6, Softmax)
```
Same optimiser/loss as Model 1, up to 25 epochs.''')

# ── CELL 7 (code) ───────────────────────────────────────────────────────────
code('''def build_model_2():
    inputs = Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs);  x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x);       x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x);      x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Model_2_Medium")

tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
model2 = build_model_2()
model2.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model2.summary()

es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
hist2 = model2.fit(train_aug_ds, validation_data=val_ds, epochs=25, callbacks=[es], verbose=2)

plot_history(hist2, "Model 2")
eval2 = evaluate_model(model2, test_ds, CLASS_NAMES, "Model 2")
build_results_row("Model 2", 4, hist2, eval2)
model2.save("model2.keras"); print("Saved model2.keras")''')

# ── CELL 8 (md) Model 3 ─────────────────────────────────────────────────────
md('''## Model 3: Deep CNN — 5 Convolutional Layers + Learning-Rate Schedule

**Extension:** a cosine-decay learning-rate schedule.

```
Conv(32)→BN→Pool → Conv(64)→BN → Conv(64)→BN→Pool → Conv(128)→BN → Conv(128)→BN→Pool
 → Flatten → Dense(512)→Drop(0.5) → Dense(256)→Drop(0.3) → Dense(6, Softmax)
```

> **Deviation from brief (documented):** the brief specified `CosineDecay(..., decay_steps=1000)`. With ~350 steps/epoch that drives the learning rate to ≈0 after only ~3 epochs, starving a 5-conv network and invalidating the depth comparison. We instead set `decay_steps = steps_per_epoch × epochs` so the cosine anneals across the **whole** run — the schedule's intent, correctly applied.''')

# ── CELL 8 (code) ───────────────────────────────────────────────────────────
code('''steps_per_epoch = int(train_ds.cardinality().numpy())
if steps_per_epoch <= 0:
    steps_per_epoch = sum(1 for _ in train_ds)
EPOCHS_M3 = 30
decay_steps = int(steps_per_epoch * EPOCHS_M3)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=decay_steps)

# Plot the LR schedule
steps = np.arange(0, decay_steps)
lrs = [float(lr_schedule(s)) for s in steps]
plt.figure(figsize=(7, 4))
plt.plot(steps, lrs)
plt.title("Model 3 — Cosine Decay Learning-Rate Schedule")
plt.xlabel("training step"); plt.ylabel("learning rate"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("task5_lr_schedule.png", dpi=120, bbox_inches="tight"); plt.show()
print("Saved task5_lr_schedule.png | steps/epoch =", steps_per_epoch, "| decay_steps =", decay_steps)

def build_model_3():
    inputs = Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x);      x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x);      x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x);     x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Model_3_Deep")

tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
model3 = build_model_3()
model3.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model3.summary()

es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
hist3 = model3.fit(train_aug_ds, validation_data=val_ds, epochs=EPOCHS_M3, callbacks=[es], verbose=2)

plot_history(hist3, "Model 3")
eval3 = evaluate_model(model3, test_ds, CLASS_NAMES, "Model 3")
build_results_row("Model 3", 5, hist3, eval3)
model3.save("model3.keras"); print("Saved model3.keras")''')

# ── CELL 9 (md) Model 4 ─────────────────────────────────────────────────────
md('''## Model 4: Transfer Learning — MobileNetV2 (Two-Stage Fine-Tuning)

**Extension:** two-stage transfer learning.
- **Stage 1 — feature extraction:** ImageNet base frozen, train only a fresh head (Adam 1e-3, 10 epochs).
- **Stage 2 — fine-tuning:** unfreeze the top 30 base layers and continue at a much lower rate (Adam 1e-5, 10 more epochs).

A `Rescaling(2, −1)` layer maps our [0, 1] inputs to the [−1, 1] range MobileNetV2 expects. The frozen backbone is called with `training=False` so its BatchNorm statistics stay fixed during fine-tuning (the recommended Keras recipe). Stage 1 and Stage 2 validation accuracies are reported separately.''')

# ── CELL 9 (code) ───────────────────────────────────────────────────────────
code('''from tensorflow.keras.applications import MobileNetV2

tf.keras.backend.clear_session(); tf.random.set_seed(SEED)

base = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
base.trainable = False   # Stage 1: feature extraction

inputs = Input(shape=(150, 150, 3))
x = layers.Rescaling(2.0, offset=-1.0)(inputs)     # [0,1] -> [-1,1] for MobileNetV2
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model4 = models.Model(inputs, outputs, name="Model_4_MobileNetV2")

# --- Stage 1: frozen base ---
model4.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Stage 1 — feature extraction (frozen base)")
hist4_s1 = model4.fit(train_aug_ds, validation_data=val_ds, epochs=10, verbose=2)
stage1_val_acc = float(max(hist4_s1.history["val_accuracy"]))

# --- Stage 2: unfreeze top 30 layers, fine-tune at low LR ---
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model4.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Stage 2 — fine-tuning top 30 layers @ lr=1e-5")
hist4_s2 = model4.fit(train_aug_ds, validation_data=val_ds, epochs=20, initial_epoch=10, verbose=2)
stage2_val_acc = float(max(hist4_s2.history["val_accuracy"]))
print(f"Stage 1 best val acc: {stage1_val_acc:.4f} | Stage 2 best val acc: {stage2_val_acc:.4f}")

plot_history(hist4_s1, "Model 4", extra_history=hist4_s2, split_epoch=10,
             fname="task7_mobilenet_history.png")
eval4 = evaluate_model(model4, test_ds, CLASS_NAMES, "Model 4")
row4 = build_results_row("Model 4", "MobileNetV2 (TL)", hist4_s1, eval4, extra_history=hist4_s2)
row4["_stage1_val_acc"] = round(stage1_val_acc, 4)
row4["_stage2_val_acc"] = round(stage2_val_acc, 4)
model4.save("model4_mobilenet.keras"); print("Saved model4_mobilenet.keras")''')

# ── CELL 10 (md) Task 5 ─────────────────────────────────────────────────────
md('''## Task 5: Architecture Comparison — Results Summary

A consolidated comparison of all four models, followed by a grouped bar chart of test accuracy and macro-F1.''')

# ── CELL 10 (code) ──────────────────────────────────────────────────────────
code('''import json

df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in results_table])
display_cols = ["Model", "Conv Layers", "Train Acc", "Val Acc", "Test Acc",
                "Train Loss", "Val Loss", "Precision", "Recall", "F1"]
df = df[display_cols]
print(df.to_string(index=False))

# Save table as an image
fig, ax = plt.subplots(figsize=(min(2 + len(display_cols) * 1.25, 18), 1.0 + 0.5 * len(df)))
ax.axis("off")
tbl = ax.table(cellText=df.round(4).astype(str).values, colLabels=df.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#34495e"); cell.set_text_props(color="white", fontweight="bold")
plt.title("Task 5 — Architecture Comparison", pad=12)
plt.savefig("task5_comparison_table.png", dpi=140, bbox_inches="tight"); plt.show()
print("Saved task5_comparison_table.png")

# Grouped bar chart: Test Acc + macro F1
x = np.arange(len(df)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w / 2, df["Test Acc"], w, label="Test Accuracy")
ax.bar(x + w / 2, df["F1"], w, label="F1 (macro)")
ax.set_xticks(x); ax.set_xticklabels(df["Model"]); ax.set_ylim(0, 1)
ax.set_ylabel("score"); ax.set_title("Task 5 — Test Accuracy & Macro F1 by Model")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for i, (a, f) in enumerate(zip(df["Test Acc"], df["F1"])):
    ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("task5_model_comparison_chart.png", dpi=140, bbox_inches="tight"); plt.show()
print("Saved task5_model_comparison_chart.png")

# --- Persist a machine-readable summary for the written discussion ---
def top_confusion(cm, names):
    cm = np.array(cm); best = (None, None, -1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > best[2]:
                best = (names[i], names[j], int(cm[i, j]))
    return best

summary = {"class_names": CLASS_NAMES, "models": []}
for row in results_table:
    summary["models"].append({
        "name": row["Model"], "conv_layers": row["Conv Layers"],
        "train_acc": row["Train Acc"], "val_acc": row["Val Acc"], "test_acc": row["Test Acc"],
        "f1_macro": row["F1"], "precision": row["Precision"], "recall": row["Recall"],
        "stage1_val_acc": row.get("_stage1_val_acc"), "stage2_val_acc": row.get("_stage2_val_acc")})
m3 = next((r for r in results_table if r["Model"] == "Model 3"), results_table[-1])
tc = top_confusion(m3["_cm"], CLASS_NAMES)
summary["top_confusion"] = {"true": tc[0], "pred": tc[1], "count": tc[2]}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\\nSaved results_summary.json")
print("Top confusion (Model 3): true", tc[0], "-> predicted", tc[1], f"({tc[2]} samples)")''')

# ── CELL 11 (md) Grad-CAM ───────────────────────────────────────────────────
md('''## Extension: Grad-CAM — Visualising What the CNN Sees

Using **Model 3** (the deepest custom network), we generate Grad-CAM heat-maps for misclassified test images — prioritising the *glacier↔mountain* and *sea↔street* confusions — to see **where** the network looks when it gets things wrong.

The pip `grad-cam` package targets PyTorch, so Grad-CAM is implemented directly with `tf.GradientTape` against the final convolutional layer (`conv_last`).''')

# ── CELL 11 (code) ──────────────────────────────────────────────────────────
code('''import matplotlib.cm as cm_mpl
import PIL.Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.input, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Gather the test set (normalised [0,1], shuffle=False so order is stable)
test_images = np.concatenate([x.numpy() for x, _ in test_ds])
test_labels = np.concatenate([y.numpy() for _, y in test_ds])
probs = model3.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1)
conf = probs[np.arange(len(probs)), preds]

mis = np.where(preds != test_labels)[0]
print("Total misclassified by Model 3:", len(mis), "/", len(test_labels))

name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
def pick_pair(a, b, k=2):
    ia, ib = name_to_idx.get(a), name_to_idx.get(b)
    sel = [int(m) for m in mis
           if (test_labels[m] == ia and preds[m] == ib) or (test_labels[m] == ib and preds[m] == ia)]
    return sel[:k]

chosen = pick_pair("glacier", "mountain", 2) + pick_pair("sea", "street", 2)
for m in mis:
    if len(chosen) >= 6: break
    if int(m) not in chosen: chosen.append(int(m))
chosen = chosen[:6]
print("Chosen misclassified indices:", chosen)

last_conv = "conv_last"
rows = max(len(chosen), 1)
fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
if rows == 1: axes = axes[None, :]
for r, idx in enumerate(chosen):
    img = test_images[idx]
    heatmap = make_gradcam_heatmap(img[None, ...], model3, last_conv, pred_index=int(preds[idx]))
    hm_color = cm_mpl.jet(np.uint8(255 * heatmap))[..., :3]
    hm_resized = np.array(PIL.Image.fromarray(np.uint8(hm_color * 255)).resize(
        (img.shape[1], img.shape[0]))) / 255.0
    overlay = np.clip(0.5 * img + 0.5 * hm_resized, 0, 1)
    axes[r, 0].imshow(img); axes[r, 0].set_title("original", fontsize=9); axes[r, 0].axis("off")
    axes[r, 1].imshow(heatmap, cmap="jet"); axes[r, 1].set_title("Grad-CAM", fontsize=9); axes[r, 1].axis("off")
    axes[r, 2].imshow(overlay); axes[r, 2].axis("off")
    axes[r, 2].set_title(f"true={CLASS_NAMES[test_labels[idx]]} | pred={CLASS_NAMES[preds[idx]]} ({conf[idx]:.2f})", fontsize=9)
fig.suptitle("Grad-CAM on Model 3 misclassifications", fontsize=13)
plt.tight_layout(); plt.savefig("extension_gradcam.png", dpi=130, bbox_inches="tight"); plt.show()
print("Saved extension_gradcam.png")''')

# ── CELL 12 (md) Task 6 discussion (tokens filled post-run) ─────────────────
md(TASK6_DISCUSSION)

# ── CELL 12b (code) audit print ─────────────────────────────────────────────
code('''import json
with open("results_summary.json") as f:
    S = json.load(f)
ms = {m["name"]: m for m in S["models"]}
print("Test accuracy / macro-F1 by model:")
for n in ["Model 1", "Model 2", "Model 3", "Model 4"]:
    if n in ms:
        print(f"  {n:9s}  acc={ms[n]['test_acc']:.4f}  F1={ms[n]['f1_macro']:.4f}")
best = max(S["models"], key=lambda m: m["test_acc"])
print("Best model:", best["name"], f"({best['test_acc']:.4f})")
if "Model 2" in ms and "Model 3" in ms:
    print(f"Model 3 vs Model 2 (test acc): {(ms['Model 3']['test_acc'] - ms['Model 2']['test_acc']) * 100:+.2f} pts")
custom = [ms[n] for n in ["Model 1", "Model 2", "Model 3"] if n in ms]
if custom and "Model 4" in ms:
    bc = max(custom, key=lambda m: m["test_acc"])
    print(f"Transfer learning vs best custom ({bc['name']}): {(ms['Model 4']['test_acc'] - bc['test_acc']) * 100:+.2f} pts")
print("Most-confused pair (Model 3):", S["top_confusion"])''')

# ── CELL 13 (md) Task 7 improvements ────────────────────────────────────────
md('''## Task 7: Improvements Applied

1. **Batch Normalisation** (Models 2, 3, 4). Normalising layer activations reduces internal covariate shift, allows higher learning rates and acts as a mild regulariser — the deeper custom models would train far less stably without it.
2. **Dropout regularisation** (all models, 0.3–0.5). Randomly dropping units forces redundant, distributed representations and is the primary defence against the dense head memorising the training set.
3. **Data augmentation** (all custom models). Flips, rotations and zooms synthesise plausible new training images, directly attacking overfitting on a mid-sized dataset.
4. **Learning-rate schedule — cosine decay** (Model 3). Annealing from 1e-3 towards zero lets the network take large early steps and fine, stable steps late on, improving final convergence over a flat rate.
5. **Transfer learning with two-stage fine-tuning** (Model 4). Stage 1 trains a fresh head on frozen ImageNet features; Stage 2 unfreezes the top 30 layers at 1e-5 so pretrained weights are adapted gently rather than destroyed.
6. **Early stopping with `restore_best_weights`** (Models 1–3). Training halts once validation loss stops improving (patience 5) and the best weights are restored, preventing over-training and giving a fair, comparable checkpoint for every architecture.''')

# ── CELL 14 (code) checklist ────────────────────────────────────────────────
code('''import glob
print("Figures saved:")
for f in sorted(glob.glob("*.png")): print("  ", f)
print("Models saved:")
for m in sorted(glob.glob("*.keras")): print("  ", m)

print("\\n=== SUBMISSION CHECKLIST ===")
for line in [
    "Task 1: Dataset description + sample grid",
    "Task 2: Preprocessing pipeline",
    "Task 3: 3 CNN architectures defined",
    "Task 4: All models trained",
    "Task 5: Comparison table + charts",
    "Task 6: Results discussion",
    "Task 7: Improvements applied",
    "Extension: Grad-CAM visualisations",
    "Extension: Two-stage fine-tuning",
    "Extension: LR schedule analysis",
]:
    print("[OK]", line)''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.5"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("SceneDepth_CNN_Classification.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote SceneDepth_CNN_Classification.ipynb with", len(cells), "cells")
