"""Bar chart of mean test accuracy per model with +/-1 s.d. error bars over the multi-seed runs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("multiseed_results.json"))
models = ["Model 1", "Model 2", "Model 3"]
means = [d[m]["test_acc_mean"] for m in models]
stds  = [d[m]["test_acc_std"] for m in models]
ns    = [d[m]["n_seeds"] for m in models]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(models))
ax.bar(x, means, yerr=stds, capsize=8,
       color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black", linewidth=0.7, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n(n={ns[i]} seeds)" for i, m in enumerate(models)])
ax.set_ylabel("Mean test accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("Multi-seed stability — mean test accuracy ± 1 s.d.\n(seeds 42, 101, 202; held-out 3,000-image test set)")
ax.grid(axis="y", alpha=0.3, zorder=0)
for i, (mn, st) in enumerate(zip(means, stds)):
    ax.text(i, mn + st + 0.02, f"{mn*100:.1f}% ± {st*100:.1f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("extension_multiseed_stability.png", dpi=300, bbox_inches="tight")
print("Saved extension_multiseed_stability.png (300 dpi)")
print("Means:", {m: f"{means[i]*100:.1f}%±{stds[i]*100:.1f}" for i, m in enumerate(models)})
order = sorted(models, key=lambda m: d[m]["test_acc_mean"])
print("Ordering (low->high):", " < ".join(order))
