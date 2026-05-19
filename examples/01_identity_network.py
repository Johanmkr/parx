"""
Example 1 — Identity network (2 → 2 → 1)
==========================================
Minimal example: a one-hidden-layer network with identity weights that
partitions R² into the four quadrants.  Demonstrates both modes, verification,
and all three visualisation functions.

Run from the repo root:
    python examples/01_identity_network.py
"""

# juliacall must be imported before torch
import parx

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from parx import compute_partition
from parx.verify import check_no_overlaps, check_covers_space
from parx.viz import plot_partition_2d, plot_region_counts, plot_halfspaces

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ── Build network ─────────────────────────────────────────────────────────────
model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1))
with torch.no_grad():
    model[0].weight.copy_(torch.eye(2))

print("Network: 2 → 2 (identity, no bias) → ReLU → 1")

# ── Sparse partition ──────────────────────────────────────────────────────────
X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
p_sparse = compute_partition(model, X, mode="sparse")
print(f"\nSparse  | regions: {len(p_sparse)}")

# ── Exact partition ───────────────────────────────────────────────────────────
x0 = np.array([1.0, 1.0])
p_exact = compute_partition(model, x0, mode="exact")
print(f"Exact   | regions: {len(p_exact)}")

# ── Verification ──────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
X_check = rng.uniform(-2, 2, (1000, 2))

ok_no_overlap, counts = check_no_overlaps(p_exact, X_check)
ok_covers,     counts = check_covers_space(p_exact, X_check)
print(f"\nVerification (exact, 1000 random points in [-2,2]²):")
print(f"  No overlaps : {'✓' if ok_no_overlap else '✗'}")
print(f"  Full cover  : {'✓' if ok_covers   else '✗'}")

# ── Visualisations ────────────────────────────────────────────────────────────
fig1 = plot_partition_2d(p_exact, x_range=(-2, 2), y_range=(-2, 2), resolution=300)
fig1.write_html(OUT / "01_partition_2d.html")
fig1.write_image(OUT / "01_partition_2d.png", scale=2)
print(f"\nSaved: {OUT / '01_partition_2d.html'}")

fig2 = plot_region_counts(p_exact)
fig2.write_html(OUT / "01_region_counts.html")
fig2.write_image(OUT / "01_region_counts.png", scale=2)
print(f"Saved: {OUT / '01_region_counts.html'}")

fig3 = plot_halfspaces(p_exact, p_exact.regions[0], x_range=(-2, 2), y_range=(-2, 2))
fig3.write_html(OUT / "01_halfspaces.html")
fig3.write_image(OUT / "01_halfspaces.png", scale=2)
print(f"Saved: {OUT / '01_halfspaces.html'}")

# ── Region details ────────────────────────────────────────────────────────────
print("\nRegions (exact):")
for i, r in enumerate(p_exact.regions):
    D, g = p_exact.halfspaces(r)
    slack = (g - D @ r.centroid).min()
    print(f"  [{i}]  centroid={np.round(r.centroid, 3)}  "
          f"constraints={D.shape[0]}  min_slack={slack:.4f}")
