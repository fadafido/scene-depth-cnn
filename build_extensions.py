"""Generate extensions.ipynb — the extra-mile analysis section.

Self-contained: reloads saved model1..4 (.keras) and recorded curves; trains TWO new
diagnostic models (Model 1-NoReg, Model 2b) without touching Models 1-4. Markdown verdicts
use __TOKENS__ filled by postprocess_ext.py from ext_results.json after execution.
"""
import json

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

# Extension E narrative — grounded in this run's param counts. Shared with postprocess_ext.py,
# which rewrites this cell in the executed notebook (param tokens filled there).
EXT_E_DISCUSSION = '''**Reading it.** Model 1 carries **__P1__** parameters (dominated by its large flattened dense layer), Model 2 **__P2__**, Model 3 **__P3__**, and MobileNetV2 (Model 4) just **__P4__**. Two things stand out. First, **MobileNetV2 is the most accurate yet by far the smallest** model — pretrained features deliver dramatically more accuracy per parameter than any custom network here. Second, *parameter count is not the same as depth*: among the custom CNNs accuracy tracks **parameters** (≈7.6M → 10.6M → 15.2M for Models 2 → 1 → 3, matching their 73% → 79% → 83% ordering), even though the four-conv Model 2 is *deeper* than the two-conv Model 1 — Model 1's flatten→dense block simply dwarfs Model 2's parameter count. So the Model 2 dip is not a raw-capacity story either; combined with Extension B (cosine LR did not rescue it), it points to a genuine training difficulty for that specific architecture. The headline holds: once a custom network is reasonably sized, only better features (transfer learning), not raw capacity, break the ~83% ceiling.'''

# ── INTRO ───────────────────────────────────────────────────────────────────
md('''# Extra-Mile Extensions — Deeper Analysis

These analyses extend the study beyond the four required models. **Models 1–4 are not retrained** — the cells below reload the saved `.keras` weights and the recorded training curves. Two *new, additional* models are trained as controlled diagnostics (**Model 1-NoReg** and **Model 2b**); neither replaces an original result. Seed 42 throughout, CPU-only, British English.''')

# ── SETUP ───────────────────────────────────────────────────────────────────
code('''%matplotlib inline
import os, random, json, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

DATA_ROOT = "data"
DATA_DIR_TRAIN = os.path.join(DATA_ROOT, "seg_train", "seg_train")
DATA_DIR_TEST  = os.path.join(DATA_ROOT, "seg_test", "seg_test")
IMG_SIZE, BATCH_SIZE = (150, 150), 32

# Rebuild the SAME pipeline as the main notebook (seed 42 -> identical 80/20 split)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="training", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="validation", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TEST, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
CLASS_NAMES = train_ds.class_names; NUM_CLASSES = len(CLASS_NAMES)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.1, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED)], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE
import shutil
CACHE_DIR = ".tfcache_ext"; shutil.rmtree(CACHE_DIR, ignore_errors=True); os.makedirs(CACHE_DIR, exist_ok=True)
def normalise_cache(ds, name):
    ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache(os.path.join(CACHE_DIR, name)).prefetch(AUTOTUNE)
train_ds = normalise_cache(train_ds, "train")
val_ds   = normalise_cache(val_ds, "val")
test_ds  = normalise_cache(test_ds, "test")
train_aug_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
steps_per_epoch = int(train_ds.cardinality().numpy())

# Reload FINAL trained models (read-only; never retrained) + recorded metadata
model1 = tf.keras.models.load_model("model1.keras")
model2 = tf.keras.models.load_model("model2.keras")
model3 = tf.keras.models.load_model("model3.keras")
model4 = tf.keras.models.load_model("model4_mobilenet.keras")
with open("results_summary.json") as f: RESULTS = json.load(f)
with open("original_histories.json") as f: ORIG_HIST = json.load(f)
RES = {m["name"]: m for m in RESULTS["models"]}
y_true = np.concatenate([y.numpy() for _, y in test_ds])   # fixed test-label order

EXT = {}
def save_ext():
    with open("ext_results.json", "w") as f: json.dump(EXT, f, indent=2)
print("Loaded models 1-4, curves, results. Classes:", CLASS_NAMES, "| steps/epoch:", steps_per_epoch)''')

# ── EXTENSION A: OVERFITTING (md) ───────────────────────────────────────────
md('''## Extension A — Overfitting, Demonstrated

*Overfitting* is when a network learns patterns specific to the training set (including noise) that do not generalise: training accuracy keeps climbing while validation accuracy stalls or falls — a widening **train–val gap**. To show this empirically we train **Model 1-NoReg**: the *same* two-conv architecture as Model 1, but with **no data augmentation and no dropout**. Everything else is identical (seed 42, 20 epochs, Adam 1e-3, early stopping on `val_loss`). The regularised Model 1 curve below is the *original* recorded run — it is not retrained.''')

# ── EXTENSION A (code) ──────────────────────────────────────────────────────
code('''def build_model_1_noreg():
    inputs = Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv_last")(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)          # NO dropout
    return models.Model(inputs, layers.Dense(NUM_CLASSES, activation="softmax")(x), name="Model_1_NoReg")

tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
model1_noreg = build_model_1_noreg()
model1_noreg.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
# NO augmentation: train on the plain normalised stream
hist_nr = model1_noreg.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[es], verbose=2)

reg = ORIG_HIST["model1"]; nr = hist_nr.history
reg_gaps = [a - v for a, v in zip(reg["accuracy"], reg["val_accuracy"])]
nr_gaps  = [a - v for a, v in zip(nr["accuracy"], nr["val_accuracy"])]
nr_test = accuracy_score(y_true, np.argmax(model1_noreg.predict(test_ds, verbose=0), axis=1))

# 2x2 comparison, shared y per row
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
er = range(1, len(reg["accuracy"]) + 1); en = range(1, len(nr["accuracy"]) + 1)
ax[0, 0].plot(er, reg["accuracy"], "o-", label="train"); ax[0, 0].plot(er, reg["val_accuracy"], "s-", label="val")
ax[0, 0].set_title("Model 1 (regularised) — accuracy"); ax[0, 0].set_ylim(0.4, 1.0)
ax[0, 1].plot(en, nr["accuracy"], "o-", label="train"); ax[0, 1].plot(en, nr["val_accuracy"], "s-", label="val")
ax[0, 1].set_title("Model 1-NoReg — accuracy"); ax[0, 1].set_ylim(0.4, 1.0)
ax[1, 0].plot(er, reg["loss"], "o-", label="train"); ax[1, 0].plot(er, reg["val_loss"], "s-", label="val")
ax[1, 0].set_title("Model 1 (regularised) — loss")
ax[1, 1].plot(en, nr["loss"], "o-", label="train"); ax[1, 1].plot(en, nr["val_loss"], "s-", label="val")
ax[1, 1].set_title("Model 1-NoReg — loss")
lmax = max(max(reg["loss"]), max(reg["val_loss"]), max(nr["loss"]), max(nr["val_loss"])) * 1.05
for a in ax.ravel(): a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
ax[1, 0].set_ylim(0, lmax); ax[1, 1].set_ylim(0, lmax)
fig.suptitle("Overfitting: regularised Model 1 vs Model 1-NoReg", fontsize=13)
plt.tight_layout(); plt.savefig("extension_overfitting_comparison.png", dpi=130, bbox_inches="tight"); plt.show()

EXT["reg_model1"] = {"final_train": reg["accuracy"][-1], "final_val": reg["val_accuracy"][-1],
                     "final_gap": reg_gaps[-1], "max_gap": max(reg_gaps)}
EXT["noreg"] = {"final_train": nr["accuracy"][-1], "final_val": nr["val_accuracy"][-1],
                "final_gap": nr_gaps[-1], "max_gap": max(nr_gaps), "test_acc": float(nr_test)}
save_ext()
print(f"Regularised Model 1 : max train-val gap {max(reg_gaps):+.3f}")
print(f"Model 1-NoReg       : max train-val gap {max(nr_gaps):+.3f} | test acc {nr_test:.4f}")
print("Saved extension_overfitting_comparison.png")''')

# ── EXTENSION A verdict (md, tokens) ────────────────────────────────────────
md('''**Result.** The regularised Model 1's train–val accuracy gap peaks at about **__REG_GAP__**, whereas Model 1-NoReg's widens to **__NOREG_GAP__**. __NOREG_VERDICT__ Model 1-NoReg's held-out test accuracy was **__NOREG_TEST__** (versus 79.3% for the regularised Model 1). Augmentation injects plausible input variation and dropout prevents units from co-adapting; together they force features that survive to the validation set — visibly narrowing the gap.''')

# ── EXTENSION B: MODEL 2b (md) ──────────────────────────────────────────────
md('''## Extension B — Diagnosing the Model 2 Dip

Model 2 (four conv layers) under-performed the two-layer Model 1 — a kept, honest result. **Was that a true depth limitation, or an optimisation artefact?** Our hypothesis: Model 2 settled in a poor basin under a flat learning rate. We test it with **Model 2b** — the *exact* Model 2 architecture plus the **cosine-decay learning-rate schedule** Model 3 uses, the only change. If 2b recovers, the dip was about optimisation; if it stays low, depth genuinely hurt here. **Model 2b is an additional diagnostic, not a replacement for the honest Model 2 result.**''')

# ── EXTENSION B (code) ──────────────────────────────────────────────────────
code('''def build_model_2():
    inputs = Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs);  x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x);       x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x);      x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    return models.Model(inputs, layers.Dense(NUM_CLASSES, activation="softmax")(x), name="Model_2b")

EPOCHS_M2B = 25
lr_sched = tf.keras.optimizers.schedules.CosineDecay(1e-3, int(steps_per_epoch * EPOCHS_M2B))
tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
model2b = build_model_2()
model2b.compile(optimizer=tf.keras.optimizers.Adam(lr_sched),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
hist2b = model2b.fit(train_aug_ds, validation_data=val_ds, epochs=EPOCHS_M2B, callbacks=[es], verbose=2)

y_pred2b = np.argmax(model2b.predict(test_ds, verbose=0), axis=1)
acc2b = accuracy_score(y_true, y_pred2b); f1_2b = f1_score(y_true, y_pred2b, average="macro")
m2 = RES["Model 2"]
EXT["model2b"] = {"test_acc": float(acc2b), "f1_macro": float(f1_2b),
                  "delta_pts": float((acc2b - m2["test_acc"]) * 100)}
save_ext()
model2b.save("model2b.keras")
print(f"Model 2  (original) : test {m2['test_acc']:.4f}  F1 {m2['f1_macro']:.4f}")
print(f"Model 2b (cosine LR): test {acc2b:.4f}  F1 {f1_2b:.4f}  | delta {(acc2b - m2['test_acc'])*100:+.1f} pts")''')

# ── EXTENSION B verdict (md, tokens) ────────────────────────────────────────
md('''**Result.** Model 2 scored 73.1% test accuracy (macro-F1 0.729). Model 2b — identical architecture, cosine learning-rate schedule added — scored **__M2B_TEST__** (F1 __M2B_F1__), a change of **__M2B_DELTA__**. __M2B_VERDICT__

The original Model 2 result stands unchanged; Model 2b only isolates *why* it happened.''')

# ── EXTENSION C: PER-CLASS METRICS (md) ─────────────────────────────────────
md('''## Extension C — Per-Class Precision, Recall and F1

Aggregate accuracy hides per-class weakness. Below, the best model (Model 4) and the baseline (Model 1) are broken down into precision, recall and F1 for each of the six classes, recomputed on the held-out test set.''')

# ── EXTENSION C (code) ──────────────────────────────────────────────────────
code('''def per_class(model):
    yp = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, yp, average=None,
                                                 labels=range(NUM_CLASSES), zero_division=0)
    return p, r, f

p1, r1, f1c = per_class(model1)
p4, r4, f4c = per_class(model4)

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
x = np.arange(NUM_CLASSES); w = 0.27
for a, (p, r, f, title) in zip(ax, [(p1, r1, f1c, "Model 1 (baseline)"),
                                    (p4, r4, f4c, "Model 4 (MobileNetV2)")]):
    a.bar(x - w, p, w, label="precision"); a.bar(x, r, w, label="recall"); a.bar(x + w, f, w, label="F1")
    a.set_xticks(x); a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    a.set_ylim(0, 1); a.set_title(title); a.legend(); a.grid(axis="y", alpha=0.3)
fig.suptitle("Per-class precision / recall / F1 (test set)", fontsize=13)
plt.tight_layout(); plt.savefig("extension_per_class_metrics.png", dpi=130, bbox_inches="tight"); plt.show()

print("Per-class F1 (Model 1 -> Model 4):")
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c:9s}: {f1c[i]:.3f} -> {f4c[i]:.3f}")
print("Saved extension_per_class_metrics.png")''')

# ── EXTENSION C discussion (md) ─────────────────────────────────────────────
md('''**Reading it.** `glacier` is the weakest class for both models — the hardest category throughout this study — routinely confused with `sea` and `mountain`. The cause is visual: glaciers share the low-texture, cool-toned, horizontally-banded look of **sea** scenes and the snow/relief of **mountains**, so the discriminative cue is small and easily missed. `street` and `buildings` are mutually confusable because a street scene contains buildings by construction, leaving framing (road foreground vs façade) as the only signal. Transfer learning (Model 4) raises every class, and most sharply the weakest ones — its ImageNet features encode the fine texture distinctions the custom networks never learn.''')

# ── EXTENSION D: GRAD-CAM CORRECT (md) ──────────────────────────────────────
md('''## Extension D — Grad-CAM on Correct Predictions

The earlier Grad-CAM examined *mistakes*. Here the same Model 3 last-conv Grad-CAM is applied to **one correctly-classified image per class**, to contrast where the network looks when it is right versus when it is wrong.''')

# ── EXTENSION D (code) ──────────────────────────────────────────────────────
code('''import matplotlib.cm as cm_mpl
import PIL.Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.input, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None: pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

test_images = np.concatenate([x.numpy() for x, _ in test_ds])
test_labels = np.concatenate([y.numpy() for _, y in test_ds])
probs3 = model3.predict(test_ds, verbose=0)
preds3 = np.argmax(probs3, axis=1); conf3 = probs3[np.arange(len(probs3)), preds3]
correct = np.where(preds3 == test_labels)[0]

chosen = []
for c in range(NUM_CLASSES):
    for idx in correct:
        if test_labels[idx] == c:
            chosen.append(int(idx)); break

fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(9, 3 * NUM_CLASSES))
for r, idx in enumerate(chosen):
    img = test_images[idx]
    hm = make_gradcam_heatmap(img[None, ...], model3, "conv_last", pred_index=int(preds3[idx]))
    hm_color = cm_mpl.jet(np.uint8(255 * hm))[..., :3]
    hm_res = np.array(PIL.Image.fromarray(np.uint8(hm_color * 255)).resize(
        (img.shape[1], img.shape[0]))) / 255.0
    overlay = np.clip(0.5 * img + 0.5 * hm_res, 0, 1)
    axes[r, 0].imshow(img); axes[r, 0].axis("off"); axes[r, 0].set_title("original", fontsize=9)
    axes[r, 1].imshow(hm, cmap="jet"); axes[r, 1].axis("off"); axes[r, 1].set_title("Grad-CAM", fontsize=9)
    axes[r, 2].imshow(overlay); axes[r, 2].axis("off")
    axes[r, 2].set_title(f"correct: {CLASS_NAMES[test_labels[idx]]} ({conf3[idx]:.2f})", fontsize=9)
fig.suptitle("Grad-CAM on CORRECT predictions (Model 3, one per class)", fontsize=13)
plt.tight_layout(); plt.savefig("extension_gradcam_correct.png", dpi=130, bbox_inches="tight"); plt.show()
print("Saved extension_gradcam_correct.png | classes:", [CLASS_NAMES[test_labels[i]] for i in chosen])''')

# ── EXTENSION D discussion (md) ─────────────────────────────────────────────
md('''**Contrast.** On correct predictions the activation tends to concentrate on the **class-defining structure** — the canopy for forest, façades for buildings/street, the water body for sea. On the earlier misclassifications it spread diffusely across the horizon band and generic texture. This matches the confusion analysis: errors arise when the discriminative region is small or shares texture with another class, so the network falls back on ambiguous global cues rather than a decisive local feature.''')

# ── EXTENSION E: COMPLEXITY (md) ────────────────────────────────────────────
md('''## Extension E — Model Complexity vs Accuracy

The research question in one figure: does spending parameters buy accuracy?''')

# ── EXTENSION E (code) ──────────────────────────────────────────────────────
code('''params = {"Model 1": model1.count_params(), "Model 2": model2.count_params(),
          "Model 3": model3.count_params(), "Model 4": model4.count_params()}
accs = {m["name"]: m["test_acc"] for m in RESULTS["models"]}

fig, ax = plt.subplots(figsize=(8.5, 6))
for name in ["Model 1", "Model 2", "Model 3", "Model 4"]:
    ax.scatter(params[name], accs[name], s=140, zorder=3)
    ax.annotate(f"{name}\\n{accs[name]*100:.1f}%  ({params[name]/1e6:.1f}M)",
                (params[name], accs[name]), textcoords="offset points", xytext=(10, -4), fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("parameters (log scale)"); ax.set_ylabel("test accuracy")
ax.set_ylim(0.6, 1.0); ax.set_title("Model complexity vs test accuracy")
ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("extension_complexity_vs_accuracy.png", dpi=140, bbox_inches="tight"); plt.show()

EXT["params"] = {k: int(v) for k, v in params.items()}
save_ext()
for name in ["Model 1", "Model 2", "Model 3", "Model 4"]:
    print(f"  {name}: {params[name]:>11,} params | test {accs[name]*100:.1f}%")
print("Saved extension_complexity_vs_accuracy.png")''')

# ── EXTENSION E discussion (md, tokens) ─────────────────────────────────────
md(EXT_E_DISCUSSION)

# ── SYNTHESIS (md) ──────────────────────────────────────────────────────────
md('''## Extensions — Synthesis

Together these diagnostics sharpen the answer to the research question:

- **Regularisation, not size, controls generalisation** — the overfitting experiment (A) shows the train–val gap is set by augmentation and dropout, not depth.
- **Optimisation vs depth are separable** — the Model 2b test (B) isolates the learning rate from the architecture.
- **Failures are class-structured** — per-class metrics (C) and Grad-CAM (D) show the custom networks break on shared-texture classes (`glacier`/`sea`/`mountain`).
- **Pretrained features win on accuracy *and* efficiency** — the complexity plot (E) shows MobileNetV2 dominating the trade-off.

The depth ceiling the custom CNNs hit is best overcome not by stacking more layers, but by importing better features through transfer learning.''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.13.5"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("extensions.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote extensions.ipynb with", len(cells), "cells")
