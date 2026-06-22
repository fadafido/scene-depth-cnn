"""Multi-seed stability run for Models 1-3 (seeds 42, 101, 202).

Seed 42 is REUSED from results_summary.json (the locked headline numbers); the saved seed-42
models are re-evaluated only to confirm they reproduce those numbers (no retraining). Seeds 101
and 202 are trained fresh as NEW model instances written to NEW files. Models 1-4 primaries are
never touched. Writes multiseed_results.json incrementally so partial progress survives a kill.

NOTE (judgement call): setting the global seed also changes the 80/20 train/val split (the
original code derives the split from `seed`). The held-out 3,000-image TEST set is identical for
every seed, so the comparison is fair; varying both initialisation and split is a stronger
stability test than varying initialisation alone.
"""
import os, sys, json, random, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR_TRAIN = "data/seg_train/seg_train"
DATA_DIR_TEST  = "data/seg_test/seg_test"
IMG, BS = (150, 150), 32
AUTOTUNE = tf.data.AUTOTUNE
NEW_SEEDS = [101, 202]
OUT_PATH = "multiseed_results.json"

def set_all_seeds(s):
    os.environ["PYTHONHASHSEED"] = str(s); random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def build_pipeline(seed):
    import shutil
    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, validation_split=0.2, subset="training", seed=seed, image_size=IMG, batch_size=BS)
    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN, validation_split=0.2, subset="validation", seed=seed, image_size=IMG, batch_size=BS)
    test = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TEST, image_size=IMG, batch_size=BS, shuffle=False)
    class_names = train.class_names
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.1, seed=seed),
        layers.RandomZoom(0.1, seed=seed)])
    cd = ".tfcache_ms"; shutil.rmtree(cd, ignore_errors=True); os.makedirs(cd, exist_ok=True)
    def nc(ds, n):
        return ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).cache(os.path.join(cd, n)).prefetch(AUTOTUNE)
    train, val, test = nc(train, "train"), nc(val, "val"), nc(test, "test")
    train_aug = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    spe = int(train.cardinality().numpy())
    return train_aug, val, test, class_names, spe

def build_model_1(n):
    i = Input((150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(i); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv_last")(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x); x = layers.Dense(128, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return models.Model(i, layers.Dense(n, activation="softmax")(x), name="Model_1")

def build_model_2(n):
    i = Input((150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(i);  x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x);  x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x); x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.4)(x)
    return models.Model(i, layers.Dense(n, activation="softmax")(x), name="Model_2")

def build_model_3(n):
    i = Input((150, 150, 3))
    x = layers.Conv2D(32, 3, activation="relu")(i); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv_last")(x); x = layers.BatchNormalization()(x); x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return models.Model(i, layers.Dense(n, activation="softmax")(x), name="Model_3")

def eval_test(model, test_ds):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))

# ── init structure + reuse seed 42 ──────────────────────────────────────────
out = {"meta": {"seeds": [42, 101, 202], "primary_seed": 42, "models": ["Model 1", "Model 2", "Model 3"],
                "std": "sample std (ddof=1)", "test_set": "fixed 3,000 held-out images",
                "note": "seed 42 reused from results_summary.json; seeds 101/202 trained fresh; "
                        "train/val split varies with seed, test set fixed"},
       "Model 1": {"seeds": {}}, "Model 2": {"seeds": {}}, "Model 3": {"seeds": {}}}
def save():
    with open(OUT_PATH, "w") as f: json.dump(out, f, indent=2)

orig = {m["name"]: m for m in json.load(open("results_summary.json"))["models"]}
for mn in ["Model 1", "Model 2", "Model 3"]:
    out[mn]["seeds"]["42"] = {"test_acc": orig[mn]["test_acc"], "f1": orig[mn]["f1_macro"], "source": "reused_seed42"}
save()

# Confirm the saved seed-42 models reproduce the headline numbers (no retraining)
print("=== Confirming saved seed-42 models reproduce headline test numbers ===", flush=True)
_, _, test42, _, _ = build_pipeline(42)
for mn, path in [("Model 1", "model1.keras"), ("Model 2", "model2.keras"), ("Model 3", "model3.keras")]:
    m = tf.keras.models.load_model(path)
    a, f = eval_test(m, test42)
    ref = orig[mn]["test_acc"]
    print(f"  {mn}: saved-model test acc {a:.4f} (f1 {f:.4f}) | results_summary {ref:.4f} | match={abs(a-ref)<0.005}", flush=True)
    out[mn]["seeds"]["42"]["reeval_check"] = {"test_acc": round(a, 4), "f1": round(f, 4)}
save()

# ── fresh seeds ─────────────────────────────────────────────────────────────
for seed in NEW_SEEDS:
    print(f"\n========== SEED {seed} ==========", flush=True)
    set_all_seeds(seed)
    train_aug, val, test, cn, spe = build_pipeline(seed)
    NUM = len(cn)
    configs = [
        ("Model 1", lambda: build_model_1(NUM), lambda: tf.keras.optimizers.Adam(1e-3), 20),
        ("Model 2", lambda: build_model_2(NUM), lambda: tf.keras.optimizers.Adam(1e-3), 25),
        ("Model 3", lambda: build_model_3(NUM),
         lambda: tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(1e-3, int(spe * 30))), 30),
    ]
    for name, mbuild, obuild, epochs in configs:
        t0 = time.time()
        try:
            tf.keras.backend.clear_session(); tf.random.set_seed(seed)
            model = mbuild()
            model.compile(optimizer=obuild(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
            hist = model.fit(train_aug, validation_data=val, epochs=epochs, callbacks=[es], verbose=2)
            acc, f1 = eval_test(model, test)
            out[name]["seeds"][str(seed)] = {"test_acc": round(acc, 4), "f1": round(f1, 4),
                                             "epochs_trained": len(hist.history["loss"])}
            fn = f"{name.replace(' ', '').lower()}_seed{seed}.keras"
            model.save(fn)
            print(f"  [{name} seed {seed}] test acc {acc:.4f} f1 {f1:.4f} | {len(hist.history['loss'])} epochs | "
                  f"{(time.time()-t0)/60:.1f} min | saved {fn}", flush=True)
        except Exception as e:
            out[name]["seeds"][str(seed)] = {"error": repr(e)}
            print(f"  [{name} seed {seed}] FAILED: {e!r}", flush=True)
        save()

# ── aggregate mean ± std ────────────────────────────────────────────────────
for mn in ["Model 1", "Model 2", "Model 3"]:
    accs = [v["test_acc"] for v in out[mn]["seeds"].values() if "test_acc" in v]
    f1s  = [v["f1"] for v in out[mn]["seeds"].values() if "f1" in v]
    out[mn]["n_seeds"] = len(accs)
    out[mn]["test_acc_mean"] = round(float(np.mean(accs)), 4) if accs else None
    out[mn]["test_acc_std"]  = round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else 0.0
    out[mn]["f1_mean"] = round(float(np.mean(f1s)), 4) if f1s else None
    out[mn]["f1_std"]  = round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0
save()

print("\n=== MULTISEED SUMMARY (mean +/- sample std) ===", flush=True)
for mn in ["Model 1", "Model 2", "Model 3"]:
    d = out[mn]
    print(f"  {mn}: acc {d['test_acc_mean']} +/- {d['test_acc_std']} | f1 {d['f1_mean']} +/- {d['f1_std']} | n={d['n_seeds']}", flush=True)
order = sorted(["Model 1", "Model 2", "Model 3"], key=lambda m: out[m]["test_acc_mean"])
print("  mean-accuracy ordering (low->high):", " < ".join(order), flush=True)
print("DONE", flush=True)
