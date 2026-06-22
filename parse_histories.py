"""Recover per-epoch training curves for Models 1-4 from the recorded papermill log,
so the overfitting comparison can show Model 1's ORIGINAL regularised curve without retraining."""
import re, json

ANSI = re.compile(r"\x1b\[[0-9;]*m")
with open("_train_log.txt") as f:
    lines = [ANSI.sub("", ln).rstrip("\n") for ln in f]

# Segment markers (substring -> segment that STARTS at this line)
def seg_for(idx, line, cur):
    if "Saved model1.keras" in line: return "_after_m1"
    if "Saved model2.keras" in line: return "_after_m2"
    if "Saved model3.keras" in line: return "_after_m3"
    if "Stage 1 —" in line: return "model4_stage1"
    if "Stage 2 —" in line: return "model4_stage2"
    if "Saved model4" in line: return "_done"
    return cur

# Walk: model1 until saved1; model2 until saved2; model3 until saved3; then stages.
cur = "model1"
buckets = {"model1": [], "model2": [], "model3": [], "model4_stage1": [], "model4_stage2": []}
metric = re.compile(r"(?<!_)accuracy:\s*([0-9.]+).*?(?<!_)loss:\s*([0-9.]+)"
                    r".*?val_accuracy:\s*([0-9.]+).*?val_loss:\s*([0-9.]+)")
state_map = {"_after_m1": "model2", "_after_m2": "model3", "_after_m3": "model3",
             "model4_stage1": "model4_stage1", "model4_stage2": "model4_stage2", "_done": "_done"}

for ln in lines:
    nxt = seg_for(0, ln, None)
    if nxt is not None:
        if nxt == "_after_m1": cur = "model2"; continue
        if nxt == "_after_m2": cur = "model3"; continue
        if nxt == "_after_m3": cur = "_gap3"; continue
        if nxt == "model4_stage1": cur = "model4_stage1"; continue
        if nxt == "model4_stage2": cur = "model4_stage2"; continue
        if nxt == "_done": cur = "_done"; continue
    m = metric.search(ln)
    if m and cur in buckets:
        buckets[cur].append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])

hist = {}
for k, rows in buckets.items():
    hist[k] = {"accuracy": [r[0] for r in rows], "loss": [r[1] for r in rows],
               "val_accuracy": [r[2] for r in rows], "val_loss": [r[3] for r in rows]}

with open("original_histories.json", "w") as f:
    json.dump(hist, f, indent=2)

for k, h in hist.items():
    n = len(h["accuracy"])
    if n:
        gap = h["accuracy"][-1] - h["val_accuracy"][-1]
        print(f"{k:15s}: {n:2d} epochs | final train {h['accuracy'][-1]:.3f} "
              f"val {h['val_accuracy'][-1]:.3f} | gap {gap:+.3f}")
    else:
        print(f"{k:15s}: 0 epochs (none parsed)")
print("Saved original_histories.json")
