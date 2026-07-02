"""Tests for parx.network.load_network."""

import pickle

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from parx.network import extract_features, load_network


def _mlp(widths: list[int]) -> nn.Sequential:
    """Build a simple ReLU MLP from a list of layer widths."""
    layers = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TestFromModule:
    def test_excludes_output_layer_by_default(self):
        model = _mlp([2, 4, 3, 2])
        weights, biases = load_network(model)
        # 3 Linear layers in total; output excluded → 2
        assert len(weights) == 2
        assert weights[0].shape == (4, 2)
        assert weights[1].shape == (3, 4)

    def test_includes_output_layer(self):
        model = _mlp([2, 4, 3, 2])
        weights, biases = load_network(model, include_output_layer=True)
        assert len(weights) == 3
        assert weights[2].shape == (2, 3)

    def test_dtype_is_float64(self):
        model = _mlp([2, 4, 2])
        weights, biases = load_network(model)
        for W, b in zip(weights, biases):
            assert W.dtype == np.float64
            assert b.dtype == np.float64

    def test_single_hidden_layer(self):
        model = _mlp([3, 5, 2])
        weights, biases = load_network(model)
        assert len(weights) == 1
        assert weights[0].shape == (5, 3)


class TestFromStateDict:
    def test_matches_module(self):
        model = _mlp([2, 4, 3, 2])
        sd = model.state_dict()
        W_mod, b_mod = load_network(model)
        W_sd, b_sd = load_network(sd)
        assert len(W_mod) == len(W_sd)
        for a, b in zip(W_mod, W_sd):
            np.testing.assert_array_almost_equal(a, b)


class TestFromPath:
    def test_load_from_pth_file(self, tmp_path):
        model = _mlp([2, 4, 3, 2])
        path = tmp_path / "model.pth"
        torch.save(model.state_dict(), path)

        W_expected, b_expected = load_network(model)
        W_loaded, b_loaded = load_network(path)

        assert len(W_loaded) == len(W_expected)
        for a, b in zip(W_expected, W_loaded):
            np.testing.assert_array_almost_equal(a, b)
        for a, b in zip(b_expected, b_loaded):
            np.testing.assert_array_almost_equal(a, b)

    def test_load_from_h5_file(self, tmp_path):
        model = _mlp([2, 4, 3, 2])
        W_expected, b_expected = load_network(model, include_output_layer=True)

        path = tmp_path / "model.h5"
        with h5py.File(path, "w") as f:
            for i, (W, b) in enumerate(zip(W_expected, b_expected)):
                f.create_dataset(f"{i}.weight", data=W)
                f.create_dataset(f"{i}.bias", data=b)

        W_loaded, b_loaded = load_network(path, include_output_layer=True)

        assert len(W_loaded) == len(W_expected)
        for a, b in zip(W_expected, W_loaded):
            np.testing.assert_array_almost_equal(a, b)
        for a, b in zip(b_expected, b_loaded):
            np.testing.assert_array_almost_equal(a, b)

    def test_load_unsupported_content_raises(self, tmp_path):
        path = tmp_path / "model.pkl"
        path.write_bytes(b"not a real checkpoint")

        with pytest.raises(pickle.UnpicklingError):
            load_network(path)


class TestExtractFeatures:
    def test_default_layer_index_uses_last_linear(self):
        model = _mlp([2, 4, 3])
        X = np.random.default_rng(0).uniform(-1, 1, (5, 2))

        features = extract_features(model, X)

        assert features.shape == (5, 4)  # input to the last Linear (4 -> 3)
        assert features.dtype == np.float64

    def test_layer_index_zero_uses_first_linear(self):
        model = _mlp([2, 4, 3])
        X = np.random.default_rng(0).uniform(-1, 1, (5, 2))

        features = extract_features(model, X, layer_index=0)

        assert features.shape == (5, 2)  # input to the first Linear (2 -> 4)

    def test_raises_on_model_without_linear_layers(self):
        model = nn.Sequential(nn.ReLU())
        X = np.zeros((3, 2))

        with pytest.raises(ValueError, match="no nn.Linear layers"):
            extract_features(model, X)
