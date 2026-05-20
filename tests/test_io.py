"""Tests for parx.io.iter_state_dicts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from parx.io import iter_state_dicts


def _mlp_state_dict(seed: int) -> dict:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).state_dict()


# ── Single state dict ────────────────────────────────────────────────────────


def test_iter_single_state_dict():
    sd = _mlp_state_dict(0)
    items = list(iter_state_dicts(sd))
    assert len(items) == 1
    label, returned = items[0]
    assert label is None
    assert returned is sd


# ── Mapping label → state_dict ───────────────────────────────────────────────


def test_iter_dict_of_state_dicts_sorted_by_key():
    src = {2: _mlp_state_dict(2), 0: _mlp_state_dict(0), 1: _mlp_state_dict(1)}
    items = list(iter_state_dicts(src))
    assert [lbl for lbl, _ in items] == [0, 1, 2]
    for lbl, sd in items:
        assert sd is src[lbl]


def test_iter_dict_of_state_dicts_string_keys():
    src = {"epoch_b": _mlp_state_dict(1), "epoch_a": _mlp_state_dict(0)}
    labels = [lbl for lbl, _ in iter_state_dicts(src)]
    assert labels == sorted(src.keys())


def test_iter_dict_rejects_non_state_dict_value():
    bad = {0: "not a state dict"}
    with pytest.raises(TypeError, match="not a state dict"):
        list(iter_state_dicts(bad))


# ── Sequence/set of state dicts ──────────────────────────────────────────────


def test_iter_list_of_state_dicts():
    sds = [_mlp_state_dict(i) for i in range(3)]
    items = list(iter_state_dicts(sds))
    assert [lbl for lbl, _ in items] == [0, 1, 2]
    for i, (_, sd) in enumerate(items):
        assert sd is sds[i]


def test_iter_tuple_of_state_dicts():
    sds = (_mlp_state_dict(0), _mlp_state_dict(1))
    items = list(iter_state_dicts(sds))
    assert len(items) == 2


# ── .h5 ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def grouped_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "epochs.h5"

    def _sd(seed):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).state_dict()

    with h5py.File(path, "w") as f:
        for epoch in (0, 1, 2):
            grp = f.create_group(str(epoch))
            for k, v in _sd(epoch).items():
                grp.create_dataset(k, data=v.numpy())
    return path


def test_iter_h5_grouped_yields_per_epoch(grouped_h5):
    items = list(iter_state_dicts(grouped_h5))
    assert [lbl for lbl, _ in items] == ["0", "1", "2"]
    for _, sd in items:
        assert any(k.endswith(".weight") for k in sd)


def test_iter_h5_flat_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "single.h5"
    torch.manual_seed(0)
    sd = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).state_dict()
    with h5py.File(path, "w") as f:
        for k, v in sd.items():
            f.create_dataset(k, data=v.numpy())

    items = list(iter_state_dicts(path))
    assert len(items) == 1
    label, returned = items[0]
    assert label is None
    assert any(k.endswith(".weight") for k in returned)


def test_iter_unsupported_path(tmp_path):
    bogus = tmp_path / "x.txt"
    bogus.write_text("nope")
    with pytest.raises(ValueError, match="Only .h5"):
        list(iter_state_dicts(bogus))
