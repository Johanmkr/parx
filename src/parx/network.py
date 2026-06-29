"""Network loading: extract (weights, biases) from PyTorch models or weight files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_network(
    model,
    *,
    include_output_layer: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract ordered (weights, biases) from a network.

    Parameters
    ----------
    model:
        ``nn.Module``, PyTorch ``state_dict`` dict, or path to a ``.pth`` /
        ``.h5`` file.
    include_output_layer:
        Include the final linear layer in the returned lists.  Defaults to
        ``False`` because only hidden ReLU layers define the polyhedral
        partition.

    Returns
    -------
    weights, biases:
        Each entry is a ``float64`` NumPy array.  Weight shape is
        ``(out_features, in_features)`` matching PyTorch convention.
    """
    if isinstance(model, str | Path):
        weights, biases = _from_path(Path(model))
    elif isinstance(model, dict):
        weights, biases = _from_state_dict(model)
    else:
        weights, biases = _from_module(model)

    if not include_output_layer and len(weights) > 1:
        weights = weights[:-1]
        biases = biases[:-1]

    return weights, biases


def extract_features(
    model,
    X: np.ndarray,
    *,
    layer_index: int = -1,
) -> np.ndarray:
    """Run a forward pass and return the input activations to a chosen Linear layer.

    ``layer_index`` uses Python list indexing over all ``nn.Linear`` submodules
    (-1 = last, 0 = first).  Returns shape ``(N, feature_dim)`` float64 array.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("torch is required: pip install torch") from exc

    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("model has no nn.Linear layers")

    target = linears[layer_index]  # natural IndexError if out of range

    captured: list = []

    def _hook(module, inputs, output):
        captured.append(inputs[0])

    handle = target.register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            model(tensor)
    finally:
        handle.remove()

    return captured[0].detach().cpu().numpy().astype(np.float64)


# ── Loaders ───────────────────────────────────────────────────────────────────


def _from_path(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if path.suffix in (".h5", ".hdf5"):
        return _from_h5(path)
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    return _from_state_dict(obj) if isinstance(obj, dict) else _from_module(obj)


def _from_h5(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for .h5 files: pip install parx[h5]"
        ) from exc

    weights, biases = [], []
    with h5py.File(path, "r") as f:
        wkeys = sorted(
            [k for k in f if "weight" in k],
            key=lambda k: int(m.group()) if (m := re.search(r"\d+", k)) else 0,
        )
        for wk in wkeys:
            bk = wk.replace("weight", "bias")
            weights.append(np.asarray(f[wk], dtype=np.float64))
            biases.append(np.asarray(f[bk], dtype=np.float64))
    return weights, biases


def _from_state_dict(
    state_dict: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    wkeys = sorted(
        [k for k in state_dict if k.endswith(".weight")],
        key=lambda k: int(m.group()) if (m := re.search(r"\d+", k)) else 0,
    )
    weights, biases = [], []
    for wk in wkeys:
        W = _as_f64(state_dict[wk])
        weights.append(W)
        bk = wk.replace(".weight", ".bias")
        if bk in state_dict:
            biases.append(_as_f64(state_dict[bk]))
        else:
            # Linear layer with bias=False — emit zeros so downstream math is uniform.
            biases.append(np.zeros(W.shape[0], dtype=np.float64))
    return weights, biases


def _from_module(model) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import torch.nn as nn

    weights, biases = [], []
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            weights.append(_as_f64(layer.weight))
            bias = (
                layer.bias if layer.bias is not None else np.zeros(layer.out_features)
            )
            biases.append(_as_f64(bias))
    return weights, biases


def _as_f64(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)
