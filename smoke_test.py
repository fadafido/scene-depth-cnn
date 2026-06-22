"""Smoke test: verify the full local pipeline + measure per-step time, before the long run."""
import os, time, random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

DATA_DIR_TRAIN = "data/seg_train/seg_train"
IMG, BS = (150, 150), 32
train = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="training", seed=SEED, image_size=IMG, batch_size=BS)
val = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN, validation_split=0.2, subset="validation", seed=SEED, image_size=IMG, batch_size=BS)
CLASS_NAMES = train.class_names; NUM = len(CLASS_NAMES)
steps_per_epoch = int(train.cardinality().numpy())
val_steps = int(val.cardinality().numpy())
print("classes:", CLASS_NAMES, "| train steps/epoch:", steps_per_epoch, "| val batches:", val_steps)

norm = tf.keras.layers.Rescaling(1/255)
AUTOTUNE = tf.data.AUTOTUNE
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.1, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED)])
train_n = train.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
train_aug = train_n.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

from tensorflow.keras import layers, models, Input
def m1():
    i = Input((150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(i); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv_last")(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x); x = layers.Dense(128, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return models.Model(i, layers.Dense(NUM, activation="softmax")(x))
def m2():
    i = Input((150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(i); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x); x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    return models.Model(i, layers.Dense(NUM, activation="softmax")(x))

N = 20
warm, sub = train_aug.take(2), train_aug.take(N)
timings = {}
for name, build in [("Model1(2-conv)", m1), ("Model2(4-conv+BN)", m2)]:
    tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
    model = build()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(warm, epochs=1, verbose=0)                 # warmup / trace
    t0 = time.time(); model.fit(sub, epochs=1, verbose=0); dt = time.time() - t0
    per_step = dt / N
    timings[name] = per_step
    print(f"{name}: {N} steps in {dt:.1f}s -> {per_step*1000:.0f} ms/step | est train epoch ~{per_step*steps_per_epoch:.0f}s")

# --- Grad-CAM path on a Keras-3 Functional model (the risky bit) ---
tf.keras.backend.clear_session()
model = m1(); model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
xb, yb = next(iter(train_n))
img = xb[:1].numpy()
gm = tf.keras.models.Model(model.inputs, [model.get_layer("conv_last").output, model.output])
with tf.GradientTape() as tape:
    co, pr = gm(img); idx = int(tf.argmax(pr[0])); cc = pr[:, idx]
g = tape.gradient(cc, co); pooled = tf.reduce_mean(g, axis=(0, 1, 2))
hm = tf.squeeze(co[0] @ pooled[..., tf.newaxis]); hm = tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)
print("Grad-CAM OK | heatmap shape:", tuple(hm.numpy().shape))

# --- save/load round-trip ---
model.save("smoke_m1.keras"); _ = tf.keras.models.load_model("smoke_m1.keras")
os.remove("smoke_m1.keras")
print("save/load OK")

# --- extrapolate full run (epochs per the brief; early stopping will usually shorten) ---
p1 = timings["Model1(2-conv)"]; p2 = timings["Model2(4-conv+BN)"]
val_overhead = 0.4  # rough val pass fraction per epoch
def est(per_step, epochs):
    return per_step * steps_per_epoch * epochs * (1 + val_overhead)
m1_t = est(p1, 20); m2_t = est(p2, 25); m3_t = est(p2*1.4, 30); m4_t = est(p2*1.3, 20)
total = (m1_t + m2_t + m3_t + m4_t) / 60
print(f"\nROUGH FULL-RUN ESTIMATE (worst case, no early stopping):")
print(f"  Model1 ~{m1_t/60:.1f} min | Model2 ~{m2_t/60:.1f} min | Model3 ~{m3_t/60:.1f} min | Model4 ~{m4_t/60:.1f} min")
print(f"  TOTAL ~{total:.0f} min ({total/60:.1f} h). Early stopping typically cuts 25-40%.")
print("SMOKE TEST PASSED")
