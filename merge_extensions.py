"""Append the executed extension cells to the main notebook, leaving Models 1-4 cells untouched."""
import sys, nbformat

MAIN = "SceneDepth_CNN_Classification.ipynb"
EXTS = sys.argv[1] if len(sys.argv) > 1 else "extensions_out.ipynb"

main = nbformat.read(MAIN, as_version=4)
exts = nbformat.read(EXTS, as_version=4)

n_before = len(main.cells)
for i, cell in enumerate(exts.cells):
    cell["id"] = f"ext-{i:02d}"          # guarantee unique, non-colliding ids
    main.cells.append(cell)

nbformat.validate(main)
nbformat.write(main, MAIN)
print(f"{MAIN}: {n_before} -> {len(main.cells)} cells (+{len(exts.cells)} extension cells)")

# sanity: no stray tokens anywhere
stray = sorted({t for c in main.cells for t in
                ["__REG_GAP__", "__NOREG_", "__M2B_", "__P1__", "__P2__", "__P3__", "__P4__"]
                if t in c.source})
print("Stray tokens in merged notebook:", stray if stray else "none")
