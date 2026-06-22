"""Three independent validation rounds for the overnight batch. Reads files on disk."""
import json, subprocess, os, re
import numpy as np

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

LOCKED_JSON = ["results_summary.json", "ext_results.json", "confusion_detail.json", "original_histories.json"]
PRIMARY_MODELS = ["model1.keras", "model2.keras", "model3.keras", "model4_mobilenet.keras"]
HEADLINE = {"Model 1": 0.793, "Model 2": 0.7313, "Model 3": 0.8273, "Model 4": 0.9023}
fails = []

# ════════════════════════════════════════════════════════════════════════════
print("=" * 70, "\nROUND 1 — Integrity of locked seed-42 artefacts\n" + "=" * 70)
# locked JSONs must be unchanged vs git (committed) — porcelain empty == clean
for f in LOCKED_JSON:
    st = git("status", "--porcelain", "--", f)
    diff = git("diff", "--stat", "--", f)
    ok = (st == "" and diff == "")
    print(f"  [{'PASS' if ok else 'FAIL'}] {f}: {'unchanged (no git diff)' if ok else 'CHANGED -> ' + st + diff}")
    if not ok: fails.append(f"R1 locked json changed: {f}")

# executed notebook + existing figures unchanged
nb_st = git("status", "--porcelain", "--", "SceneDepth_CNN_Classification.ipynb")
print(f"  [{'PASS' if nb_st=='' else 'FAIL'}] executed notebook unchanged: {'yes' if nb_st=='' else nb_st}")
if nb_st: fails.append("R1 notebook changed")
mod_png = [l for l in git("status", "--porcelain", "--", "*.png").splitlines() if l.startswith(" M") or l.startswith("M")]
print(f"  [{'PASS' if not mod_png else 'FAIL'}] no existing figure modified: {'none modified' if not mod_png else mod_png}")
if mod_png: fails.append("R1 figure modified")

# primary models not retrained (untracked/gitignored; mtimes predate tonight's seed run)
print("  primary model files (must predate the multi-seed run; only READ, never rewritten):")
seed_mtime = os.path.getmtime("multiseed_results.json")
for m in PRIMARY_MODELS:
    if os.path.exists(m):
        mt = os.path.getmtime(m)
        older = mt < seed_mtime - 1
        print(f"     [{'PASS' if older else 'FAIL'}] {m}: mtime {'older than multiseed run' if older else 'NEWER — possible overwrite!'}")
        if not older: fails.append(f"R1 primary model touched: {m}")
    else:
        print(f"     [FAIL] {m}: MISSING"); fails.append(f"R1 missing primary: {m}")
print("  new seed model files (expected, gitignored):", sorted(f for f in os.listdir(".") if re.match(r"model[123]_seed", f)))

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, "\nROUND 2 — Correctness of new outputs\n" + "=" * 70)
ms = json.load(open("multiseed_results.json"))
print("  recomputed mean/std from per-seed values vs stored:")
for mn in ["Model 1", "Model 2", "Model 3"]:
    seeds = ms[mn]["seeds"]
    accs = [seeds[s]["test_acc"] for s in ["42", "101", "202"] if "test_acc" in seeds.get(s, {})]
    f1s  = [seeds[s]["f1"] for s in ["42", "101", "202"] if "f1" in seeds.get(s, {})]
    rec_am, rec_as = round(float(np.mean(accs)), 4), round(float(np.std(accs, ddof=1)), 4)
    rec_fm, rec_fs = round(float(np.mean(f1s)), 4), round(float(np.std(f1s, ddof=1)), 4)
    ok = (rec_am == ms[mn]["test_acc_mean"] and rec_as == ms[mn]["test_acc_std"]
          and rec_fm == ms[mn]["f1_mean"] and rec_fs == ms[mn]["f1_std"])
    print(f"     [{'PASS' if ok else 'FAIL'}] {mn}: acc {rec_am}±{rec_as} (stored {ms[mn]['test_acc_mean']}±{ms[mn]['test_acc_std']}) "
          f"| f1 {rec_fm}±{rec_fs} | n={ms[mn]['n_seeds']} | seeds {accs}")
    if not ok: fails.append(f"R2 stat mismatch {mn}")

fig = "extension_multiseed_stability.png"
ok = os.path.exists(fig) and os.path.getsize(fig) > 0
print(f"  [{'PASS' if ok else 'FAIL'}] {fig} exists & non-zero: {os.path.getsize(fig) if ok else 'NO'} bytes")
if not ok: fails.append("R2 figure missing")

rs = {m["name"]: m for m in json.load(open("results_summary.json"))["models"]}
for mn, exp in HEADLINE.items():
    got = rs[mn]["test_acc"]
    ok = (got == exp)
    print(f"  [{'PASS' if ok else 'FAIL'}] headline {mn} test_acc in results_summary.json = {got} (expect {exp})")
    if not ok: fails.append(f"R2 headline drift {mn}")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70, "\nROUND 3 — Repo consistency & reproducibility\n" + "=" * 70)
req = [l.strip() for l in open("requirements.txt") if l.strip() and not l.strip().startswith("#")]
bad = [l for l in req if not re.match(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9,_\-]+\])?([<>=!~]=?[0-9].*)?(\s*#.*)?$", l)]
ok = len(req) >= 10 and not bad
print(f"  [{'PASS' if ok else 'FAIL'}] requirements.txt: {len(req)} deps, format valid: {'yes' if ok else 'BAD: '+str(bad)}")
if not ok: fails.append("R3 requirements invalid")

readme = open("README.md").read()
print("  README multi-seed numbers vs multiseed_results.json:")
for mn in ["Model 1", "Model 2", "Model 3"]:
    m, s = ms[mn]["test_acc_mean"], ms[mn]["test_acc_std"]
    token = f"{m*100:.1f}% ± {s*100:.1f}"
    ok = token in readme
    print(f"     [{'PASS' if ok else 'FAIL'}] '{token}' present in README: {ok}")
    if not ok: fails.append(f"R3 README drift {mn} ({token})")

gi = open(".gitignore").read()
for pat in ["data/", "*.keras", ".tfcache"]:
    ok = pat in gi
    print(f"  [{'PASS' if ok else 'FAIL'}] .gitignore excludes '{pat}': {ok}")
    if not ok: fails.append(f"R3 gitignore missing {pat}")

# stage everything (respects .gitignore) then list files that will be pushed + sizes
subprocess.run(["git", "add", "-A"], capture_output=True)
tracked = git("ls-files").splitlines()
print(f"  files tracked/to-be-pushed: {len(tracked)}")
big = []
for f in tracked:
    if os.path.exists(f):
        mb = os.path.getsize(f) / 1e6
        if mb > 100: big.append((f, mb))
print(f"  [{'PASS' if not big else 'FAIL'}] no tracked file > 100 MB (GitHub limit): {'none' if not big else big}")
if big: fails.append("R3 oversized file")
total_mb = sum(os.path.getsize(f) for f in tracked if os.path.exists(f)) / 1e6
print(f"  total tracked size: {total_mb:.1f} MB")
# confirm no .keras / data staged
leaked = [f for f in tracked if f.endswith(".keras") or f.startswith("data/") or f.startswith(".tfcache")]
print(f"  [{'PASS' if not leaked else 'FAIL'}] no weights/data/cache staged: {'clean' if not leaked else leaked}")
if leaked: fails.append("R3 leaked heavyweight")

print("\n" + "=" * 70)
print("VALIDATION RESULT:", "ALL PASS ✔" if not fails else f"{len(fails)} FAILURE(S): {fails}")
print("=" * 70)
