"""Fill Extension verdict tokens from ext_results.json + results_summary.json."""
import json, sys, nbformat

INP = sys.argv[1] if len(sys.argv) > 1 else "extensions_out.ipynb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "extensions_out.ipynb"

EXT = json.load(open("ext_results.json"))
RES = {m["name"]: m for m in json.load(open("results_summary.json"))["models"]}

reg, nr = EXT["reg_model1"], EXT["noreg"]
m2b, params = EXT["model2b"], EXT["params"]
m1_test, m2_test = RES["Model 1"]["test_acc"], RES["Model 2"]["test_acc"]

# Overfitting verdict
gap_diff = nr["max_gap"] - reg["max_gap"]
if gap_diff >= 0.05:
    noreg_verdict = ("This markedly wider gap is overfitting in action: stripped of augmentation "
                     "and dropout, the network fits the training set far more tightly than it generalises.")
elif gap_diff >= 0.02:
    noreg_verdict = ("This wider gap is consistent with overfitting — the unregularised network "
                     "generalises less well than its regularised twin.")
else:
    noreg_verdict = ("The gap did not widen as clearly as expected in this run; we report the curves "
                     "honestly rather than force the narrative.")

# Model 2b verdict
d = m2b["delta_pts"]
if d >= 3:
    beat = " It now matches or exceeds the two-layer baseline." if m2b["test_acc"] >= m1_test else ""
    m2b_verdict = ("Model 2b recovered substantially — the dip was largely an optimisation/learning-rate "
                   "artefact: with a cosine schedule the same four-layer architecture trains to a much "
                   "better basin, so added depth was not the core problem." + beat)
elif d >= 1:
    m2b_verdict = ("Model 2b improved only modestly — the learning rate explains part of the dip but not "
                   "all of it; some of the four-layer model's weakness is genuine.")
else:
    m2b_verdict = ("Model 2b did not recover — the four-layer model underperforms here regardless of the "
                   "learning-rate schedule, so the dip reflects a real training difficulty for this "
                   "architecture rather than merely the optimiser. A valid, kept finding. It also clarifies "
                   "Model 3's recovery: since the cosine schedule alone does not lift Model 2, Model 3's "
                   "gain owes to its added capacity and larger head, not the learning rate.")

repl = {
    "__REG_GAP__": f"{reg['max_gap']*100:.1f} points",
    "__NOREG_GAP__": f"{nr['max_gap']*100:.1f} points",
    "__NOREG_VERDICT__": noreg_verdict,
    "__NOREG_TEST__": f"{nr['test_acc']*100:.1f}%",
    "__M2B_TEST__": f"{m2b['test_acc']*100:.1f}%",
    "__M2B_F1__": f"{m2b['f1_macro']:.3f}",
    "__M2B_DELTA__": f"{m2b['delta_pts']:+.1f} percentage points",
    "__M2B_VERDICT__": m2b_verdict,
    "__P1__": f"{params['Model 1']:,}",
    "__P2__": f"{params['Model 2']:,}",
    "__P3__": f"{params['Model 3']:,}",
    "__P4__": f"{params['Model 4']:,}",
}

# Corrected Extension E narrative (param tokens filled below).
EXT_E_DISCUSSION = '''**Reading it.** Model 1 carries **__P1__** parameters (dominated by its large flattened dense layer), Model 2 **__P2__**, Model 3 **__P3__**, and MobileNetV2 (Model 4) just **__P4__**. Two things stand out. First, **MobileNetV2 is the most accurate yet by far the smallest** model — pretrained features deliver dramatically more accuracy per parameter than any custom network here. Second, *parameter count is not the same as depth*: among the custom CNNs accuracy tracks **parameters** (≈7.6M → 10.6M → 15.2M for Models 2 → 1 → 3, matching their 73% → 79% → 83% ordering), even though the four-conv Model 2 is *deeper* than the two-conv Model 1 — Model 1's flatten→dense block simply dwarfs Model 2's parameter count. So the Model 2 dip is not a raw-capacity story either; combined with Extension B (cosine LR did not rescue it), it points to a genuine training difficulty for that specific architecture. The headline holds: once a custom network is reasonably sized, only better features (transfer learning), not raw capacity, break the ~83% ceiling.'''

nb = nbformat.read(INP, as_version=4)
# Replace the Extension E discussion cell wholesale (its premise changed with the real param counts)
for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.lstrip().startswith("**Reading it.** Model 1 carries"):
        cell.source = EXT_E_DISCUSSION
# Token fill across all markdown cells
for cell in nb.cells:
    if cell.cell_type == "markdown":
        for k, v in repl.items():
            cell.source = cell.source.replace(k, v)
nbformat.write(nb, OUT)

leftover = sorted({t for c in nb.cells for t in repl if t in c.source})
print("Filled extension tokens:")
for k, v in repl.items():
    print(f"  {k:18s} -> {v[:70]}")
print("Leftover tokens:", leftover if leftover else "none")
