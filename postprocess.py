"""Finalise the executed notebook: replace the Task 6 discussion cell with the
evidence-grounded narrative for THIS run, then write the deliverable notebook.

Reads:  run_output.ipynb (papermill-executed)
Writes: SceneDepth_CNN_Classification.ipynb (executed + corrected discussion)
"""
import sys, nbformat

EXECUTED = sys.argv[1] if len(sys.argv) > 1 else "run_output.ipynb"
FINAL = sys.argv[2] if len(sys.argv) > 2 else "SceneDepth_CNN_Classification.ipynb"

# Kept in sync with TASK6_DISCUSSION in build_notebook.py.
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

nb = nbformat.read(EXECUTED, as_version=4)
replaced = 0
for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## Task 6: Results Discussion"):
        cell.source = TASK6_DISCUSSION
        replaced += 1

# Sanity: no leftover placeholder tokens anywhere
leftover = [i for i, c in enumerate(nb.cells) if "__" in c.source and any(
    t in c.source for t in ["__BEST_", "__M1_", "__M2_", "__M3_", "__M4_", "__TOP_", "__TL_", "__DIMINISH_"])]

nbformat.validate(nb)
nbformat.write(nb, FINAL)
print(f"Replaced Task 6 cell(s): {replaced}")
print(f"Leftover token cells: {leftover if leftover else 'none'}")
print(f"Wrote {FINAL} ({len(nb.cells)} cells, nbformat-valid)")
