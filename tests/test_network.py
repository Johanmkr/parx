"""Tests for parx.network.load_network."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from parx.network import load_network


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
