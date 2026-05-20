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
from parx.viz import plot_partition_2d

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


# ── Visualise the partition at the first vs last snapshot ────────────────────
first_epoch = min(snapshots)
last_epoch  = max(snapshots)

DOMAIN = ((-1.5, 1.5), (-1.5, 1.5))
p_first = compute_partition(snapshots[first_epoch], np.zeros(2), method="exact_julia_fast")
p_last  = compute_partition(snapshots[last_epoch],  np.zeros(2), method="exact_julia_fast")

fig_first = plot_partition_2d(p_first, domain=DOMAIN)
fig_first.update_layout(title=f"Epoch {first_epoch}: {len(p_first)} regions  (‖A‖_F)")
_save(fig_first, f"03_partition_epoch{first_epoch:02d}")

fig_last  = plot_partition_2d(p_last, domain=DOMAIN)
fig_last.update_layout(title=f"Epoch {last_epoch}: {len(p_last)} regions  (‖A‖_F)")
_save(fig_last, f"03_partition_epoch{last_epoch:02d}")

print("\nDone.")
