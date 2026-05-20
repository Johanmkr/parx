"""
Example 2 — Random MLP  (2 → 5 → 5 → 5 → 1)
=============================================
Demonstrates:

* state-dict workflow: train/load → ``state_dict`` → ``compute_partition``
* the method registry: sparse vs exact, Julia vs Python implementations
* colour-by-metric plotting (``‖A‖_F`` of the local affine map)

Run from the repo root:
    python examples/02_random_mlp.py
"""

# juliacall must be imported before torch
import parx

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from parx import compute_partition, list_methods, precompile
from parx.verify import check_no_overlaps, check_covers_space
from parx.viz import (
    plot_partition_2d,
    plot_region_counts,
    affine_frobenius,
    affine_spectral,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

SEED  = 42
RANGE = (-1.0, 1.0)
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


# ── Build network and grab its state dict ────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 5), nn.ReLU(),
    nn.Linear(5, 5), nn.ReLU(),
    nn.Linear(5, 5), nn.ReLU(),
    nn.Linear(5, 1),
)
state_dict = model.state_dict()
print("Network: 2 → 5 → 5 → 5 → 1  (random weights, seed=42)")
print(f"state_dict keys: {list(state_dict)}")
print(f"\nRegistered methods: {list_methods()}")

# ── Sparse partition (from state_dict) ───────────────────────────────────────
X_train = rng.uniform(*RANGE, (500, 2))
p_sparse = compute_partition(state_dict, X_train, method="sparse_julia")
print(f"\nSparse  | regions: {len(p_sparse)}  (500 points in {RANGE})")

# ── Exact partition — fast Julia path ────────────────────────────────────────
x0 = np.zeros(2)
p_exact = compute_partition(state_dict, x0, method="exact_julia_fast")
print(f"Exact   | regions: {len(p_exact)}")
print(f"        | {len(p_exact) - len(p_sparse):+d} more regions than sparse")

# ── Cross-method timing ──────────────────────────────────────────────────────
# precompile() amortises Julia's TTFX (time-to-first-X) so the timings below
# reflect actual work, not JIT compilation.  Skip this when you just want one
# region computation — the cost is paid once per Python process either way.
print("\nWarming up Julia (precompile)…")
precompile(verbose=True)

print("\nExact-mode timing (same network, same x0):")
for m in ("exact_julia", "exact_julia_fast", "exact_python"):
    t0 = time.perf_counter()
    p = compute_partition(state_dict, x0, method=m)
    dt = time.perf_counter() - t0
    print(f"  {m:<20} {len(p):>4} regions   {dt:.2f}s")

# ── Verification ─────────────────────────────────────────────────────────────
X_check = rng.uniform(*RANGE, (2000, 2))
ok_no_overlap, _      = check_no_overlaps(p_exact, X_check)
ok_covers, counts     = check_covers_space(p_exact, X_check)
print(f"\nVerification (exact_julia_fast, 2000 random points in {RANGE}):")
print(f"  No overlaps : {'OK' if ok_no_overlap else 'FAIL'}")
print(f"  Full cover  : {'OK' if ok_covers    else 'FAIL'}")

# ── Figures ──────────────────────────────────────────────────────────────────
print("\nFigures (opening in browser):")

fig_sparse = plot_partition_2d(p_sparse, domain=DOMAIN)  # default color_by=affine_frobenius
fig_sparse.update_layout(title=f"Sparse partition  ({len(p_sparse)} regions, ‖A‖_F)")
fig_sparse.show()
save(fig_sparse, "02_sparse_partition")

fig_exact = plot_partition_2d(p_exact, domain=DOMAIN)
fig_exact.update_layout(title=f"Exact partition  ({len(p_exact)} regions, ‖A‖_F)")
fig_exact.show()
save(fig_exact, "02_exact_partition")

# Spectral norm with log colour
fig_spec = plot_partition_2d(
    p_exact, domain=DOMAIN, color_by=affine_spectral, log_color=True, colorscale="Plasma"
)
fig_spec.update_layout(title=f"Exact partition  (spectral norm, log scale)")
fig_spec.show()
save(fig_spec, "02_exact_spectral")

fig_counts = plot_region_counts(p_exact)
fig_counts.show()
save(fig_counts, "02_region_counts")
