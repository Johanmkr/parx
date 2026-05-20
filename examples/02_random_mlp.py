"""
Example 2 — Random MLP  (2 → 5 → 5 → 5 → 1)
=============================================
Compares sparse and exact modes on a randomly initialised network.
Runs verification and saves figures.

Run from the repo root:
    python examples/02_random_mlp.py

Plots open automatically in the browser via fig.show().
HTML copies are also saved to examples/output/ for later reference.
PNG export requires Chrome once (run: plotly_get_chrome).
"""

# juliacall must be imported before torch
import parx

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from parx import compute_partition
from parx.verify import check_no_overlaps, check_covers_space
from parx.viz import plot_partition_2d, plot_region_counts

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

SEED = 42
RANGE = (-1.0, 1.0)   # training data domain
DOMAIN = (RANGE, RANGE)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def save(fig, stem: str) -> None:
    html = OUT / f"{stem}.html"
    fig.write_html(html)
    print(f"  Saved: {html}")
    try:
        png = OUT / f"{stem}.png"
        fig.write_image(png, scale=2)
        print(f"  Saved: {png}")
    except Exception:
        print(f"  (PNG skipped — run 'plotly_get_chrome' once to enable)")


# ── Build network ─────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 5), nn.ReLU(),
    nn.Linear(5, 5), nn.ReLU(),
    nn.Linear(5, 5), nn.ReLU(),
    nn.Linear(5, 1),
)
print("Network: 2 → 5 → 5 → 5 → 1  (random weights, seed=42)")

# ── Sparse partition ──────────────────────────────────────────────────────────
X_train = rng.uniform(*RANGE, (500, 2))
p_sparse = compute_partition(model, X_train, mode="sparse")
print(f"\nSparse  | regions: {len(p_sparse)}  (500 points in {RANGE})")

# ── Exact partition ───────────────────────────────────────────────────────────
x0 = np.zeros(2)
p_exact = compute_partition(model, x0, mode="exact")
print(f"Exact   | regions: {len(p_exact)}")
print(f"        | {len(p_exact) - len(p_sparse):+d} more regions than sparse")

# ── Verification (exact mode) ─────────────────────────────────────────────────
X_check = rng.uniform(*RANGE, (2000, 2))
ok_no_overlap, _ = check_no_overlaps(p_exact, X_check)
ok_covers, counts = check_covers_space(p_exact, X_check)
print(f"\nVerification (exact, 2000 random points in {RANGE}):")
print(f"  No overlaps : {'OK' if ok_no_overlap else 'FAIL'}")
print(f"  Full cover  : {'OK' if ok_covers    else 'FAIL'}")
if not ok_no_overlap:
    print(f"  Max membership count : {counts.max()}")
if not ok_covers:
    print(f"  Uncovered: {(counts==0).sum()}  |  Overlapping: {(counts>1).sum()}")

# ── Routing coverage ─────────────────────────────────────────────────────────
routed = p_sparse.route(X_train)
n_covered = sum(r is not None for r in routed)
print(f"\nSparse routing: {n_covered}/{len(X_train)} training points covered "
      f"({100 * n_covered / len(X_train):.1f} %)")

# ── Figures ───────────────────────────────────────────────────────────────────
print("\nFigures (opening in browser):")

fig_sparse = plot_partition_2d(p_sparse, domain=DOMAIN)
fig_sparse.update_layout(title=f"Sparse partition  ({len(p_sparse)} regions)")
fig_sparse.show()
save(fig_sparse, "02_sparse_partition")

fig_exact = plot_partition_2d(p_exact, domain=DOMAIN)
fig_exact.update_layout(title=f"Exact partition  ({len(p_exact)} regions)")
fig_exact.show()
save(fig_exact, "02_exact_partition")

fig_counts = plot_region_counts(p_exact)
fig_counts.show()
save(fig_counts, "02_region_counts")
