"""
Example 3 — Partition evolution over a full training run
=========================================================

Trains a small network to convergence, capturing a state-dict snapshot at
every epoch.  Computes the exact linear partition for each snapshot, then
produces:

  * a static plot of the initial (random) and final (converged) partition
  * an interactive Plotly animation with a play button and epoch slider
  * an animated GIF (requires ``pip install 'parx[animate]'``)

Run from the repo root:
    python examples/03_epochs.py
"""

# juliacall must be imported before torch
import parx

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from parx import compute_partition
from parx.viz import animate_epochs, animate_epochs_video, plot_partition_2d

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(0)

DOMAIN_RANGE = (-1.5, 1.5)


def _save(fig, stem: str) -> None:
    html = OUT / f"{stem}.html"
    fig.write_html(html)
    print(f"  Saved: {html}")


# ── Data: points inside / outside the unit circle ────────────────────────────
def _make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = rng.uniform(DOMAIN_RANGE[0], DOMAIN_RANGE[1], (n, 2)).astype(np.float32)
    y = (np.linalg.norm(x, axis=1) < 1.0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


X_train, y_train = _make_data(512)

model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1), nn.Sigmoid(),
)
opt    = torch.optim.Adam(model.parameters(), lr=0.03)
loss_f = nn.BCELoss()

# ── Training loop — snapshot every epoch ─────────────────────────────────────
N_EPOCHS = 120
snapshots: dict[int, dict] = {}

print(f"Training for {N_EPOCHS} epochs …")
print(f"{'Epoch':>6}  {'Loss':>8}")
for epoch in range(N_EPOCHS + 1):
    snapshots[epoch] = copy.deepcopy(model.state_dict())
    if epoch % 10 == 0:
        with torch.no_grad():
            loss_val = loss_f(model(X_train).squeeze(-1), y_train).item()
        print(f"{epoch:>6}  {loss_val:>8.4f}")
    if epoch < N_EPOCHS:
        opt.zero_grad()
        loss_f(model(X_train).squeeze(-1), y_train).backward()
        opt.step()

print(f"\nCaptured {len(snapshots)} snapshots (epochs 0 – {N_EPOCHS})")

# ── Compute exact partition for every epoch ───────────────────────────────────
# exact_julia_fast uses DFS + facet-flipping seeded from the origin; fast for
# small 2-D networks (typically < 0.5 s per partition).
print(f"\nComputing exact partitions for all {len(snapshots)} epochs …")
epoch_keys = sorted(snapshots)
partitions = []
for epoch in epoch_keys:
    p = compute_partition(snapshots[epoch], np.zeros(2), method="exact_julia_fast")
    partitions.append(p)
    if epoch % 20 == 0:
        print(f"  epoch {epoch:>3}: {len(p):>3} regions")

print(f"  … done.  Region counts range "
      f"{min(len(p) for p in partitions)} – {max(len(p) for p in partitions)}")

epoch_labels = [str(e) for e in epoch_keys]

# ── Static comparison: epoch 0 vs final ──────────────────────────────────────
print("\nSaving static plots …")
for idx, label in [(0, "initial"), (-1, "final")]:
    fig = plot_partition_2d(
        partitions[idx],
        x_range=DOMAIN_RANGE,
        y_range=DOMAIN_RANGE,
        colorscale="Plasma",
    )
    epoch_num = epoch_keys[idx]
    fig.update_layout(
        title=f"Epoch {epoch_num} ({label}): {len(partitions[idx])} regions  (‖A‖_F)"
    )
    _save(fig, f"03_partition_{label}")

# ── Interactive Plotly animation (play/pause + epoch slider) ──────────────────
# animate_epochs returns a go.Figure with one frame per epoch.
# All frames share a fixed spatial range and colour scale so that the growth
# of regions during training is visually comparable across time.
# The returned figure can be shown in a notebook with .show(), saved as a
# self-contained interactive HTML, or embedded in a dashboard.
print("\nBuilding interactive Plotly animation …")
fig_anim = animate_epochs(
    partitions,
    epoch_labels=epoch_labels,
    x_range=DOMAIN_RANGE,
    y_range=DOMAIN_RANGE,
    colorscale="Plasma",
    frame_duration=120,     # ms per frame — fast enough to see convergence
)
fig_anim.show()
_save(fig_anim, "03_animation")

# ── GIF export via matplotlib ─────────────────────────────────────────────────
# animate_epochs_video writes a .gif (Pillow writer) or .mp4 (ffmpeg writer)
# depending on the file suffix.  Requires: pip install 'parx[animate]'
try:
    gif_path = OUT / "03_animation.gif"
    print("Building GIF …")
    animate_epochs_video(
        partitions,
        gif_path,
        epoch_labels=epoch_labels,
        x_range=DOMAIN_RANGE,
        y_range=DOMAIN_RANGE,
        colorscale="Plasma",
        fps=12,             # smooth enough to follow convergence
        dpi=120,
    )
    print(f"  Saved: {gif_path}")
except ImportError as e:
    print(f"  GIF skipped — {e}")

print("\nDone.")
