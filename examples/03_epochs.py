"""
Example 3 — Per-epoch partitions over training
==============================================

Demonstrates the state-dict-per-epoch workflow:

  1. train a small net for a few epochs, capturing state dicts as we go
  2. drive :func:`parx.iter_state_dicts` over the snapshots
  3. compute one partition per epoch and watch the region count grow

The same loop works unchanged if you pass:

  * a ``dict[label, state_dict]``  (in-memory snapshots)
  * a ``.h5`` file with one group per epoch
  * a flat sequence of state dicts

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

from parx import compute_partition, iter_state_dicts
from parx.viz import animate_epochs, animate_epochs_video, plot_partition_2d

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
rng = np.random.default_rng(0)


def _save(fig, stem: str) -> None:
    html = OUT / f"{stem}.html"
    fig.write_html(html)
    print(f"  Saved: {html}")


# ── Toy training loop on a circle-vs-square classification task ──────────────
def _make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = rng.uniform(-1.5, 1.5, (n, 2)).astype(np.float32)
    r = np.linalg.norm(x, axis=1)
    y = (r < 1.0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


X_train, y_train = _make_data(512)
X_eval = rng.uniform(-1.5, 1.5, (200, 2))

model = nn.Sequential(
    nn.Linear(2, 6), nn.ReLU(),
    nn.Linear(6, 6), nn.ReLU(),
    nn.Linear(6, 1), nn.Sigmoid(),
)
opt    = torch.optim.Adam(model.parameters(), lr=0.05)
loss_f = nn.BCELoss()

# Capture snapshots every few epochs.  ``copy.deepcopy`` is required because
# state_dict() returns views into the live model parameters.
snapshots: dict[int, dict] = {0: copy.deepcopy(model.state_dict())}
for epoch in range(1, 21):
    opt.zero_grad()
    pred = model(X_train).squeeze(-1)
    loss = loss_f(pred, y_train)
    loss.backward()
    opt.step()
    if epoch % 5 == 0:
        snapshots[epoch] = copy.deepcopy(model.state_dict())

print(f"Captured snapshots at epochs: {sorted(snapshots)}")


# ── Compute one partition per snapshot ───────────────────────────────────────
print("\nEpoch | sparse regions | exact regions")
print("------+----------------+--------------")
for epoch, sd in iter_state_dicts(snapshots):
    p_sparse = compute_partition(sd, X_eval, method="sparse_julia")
    p_exact  = compute_partition(sd, np.zeros(2), method="exact_julia_fast")
    print(f"{epoch:>5} | {len(p_sparse):>14} | {len(p_exact):>13}")


# ── Compute exact partitions for all snapshots ───────────────────────────────
DOMAIN = ((-1.5, 1.5), (-1.5, 1.5))

print("\nComputing exact partitions for all snapshots …")
epoch_keys = sorted(snapshots)
partitions = []
for epoch in epoch_keys:
    p = compute_partition(snapshots[epoch], np.zeros(2), method="exact_julia_fast")
    partitions.append(p)
    print(f"  epoch {epoch:>2}: {len(p)} regions")

epoch_labels = [str(e) for e in epoch_keys]

# ── Static plots for first and last snapshot ─────────────────────────────────
fig_first = plot_partition_2d(partitions[0], domain=DOMAIN)
fig_first.update_layout(title=f"Epoch {epoch_keys[0]}: {len(partitions[0])} regions  (‖A‖_F)")
_save(fig_first, f"03_partition_epoch{epoch_keys[0]:02d}")

fig_last = plot_partition_2d(partitions[-1], domain=DOMAIN)
fig_last.update_layout(title=f"Epoch {epoch_keys[-1]}: {len(partitions[-1])} regions  (‖A‖_F)")
_save(fig_last, f"03_partition_epoch{epoch_keys[-1]:02d}")

# ── Interactive Plotly animation (play/pause + epoch slider) ─────────────────
# animate_epochs returns a go.Figure with one frame per epoch.
# All frames share the same spatial range and colour scale so changes across
# training are visually comparable.  Call .show() to open in a browser, or
# write_html() to save a self-contained interactive file.
print("\nBuilding interactive Plotly animation …")
fig_anim = animate_epochs(
    partitions,
    epoch_labels=epoch_labels,
    x_range=(-1.5, 1.5),   # fix view to the training domain
    y_range=(-1.5, 1.5),
    colorscale="Plasma",
    frame_duration=700,     # ms per frame during auto-play
)
fig_anim.show()
_save(fig_anim, "03_animation")

# ── GIF export via matplotlib ─────────────────────────────────────────────────
# animate_epochs_video writes a .gif (Pillow) or .mp4 (ffmpeg) depending on
# the suffix.  Requires: pip install 'parx[animate]'
try:
    gif_path = OUT / "03_animation.gif"
    print("Building GIF (matplotlib FuncAnimation) …")
    animate_epochs_video(
        partitions,
        gif_path,
        epoch_labels=epoch_labels,
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        colorscale="Plasma",
        fps=2,
        dpi=120,
    )
    print(f"  Saved: {gif_path}")
except ImportError as e:
    print(f"  GIF skipped — {e}")

print("\nDone.")
