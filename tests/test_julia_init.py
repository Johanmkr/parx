"""Tests for parx._julia_init._resolve_juliaup_shim (pure Python, no Julia needed)."""

import json
import os
import stat

from parx._julia_init import _resolve_juliaup_shim


def _make_shim(bin_dir):
    """Create a fake juliaup launcher shim: julia -> julialauncher."""
    os.makedirs(bin_dir, exist_ok=True)
    launcher = os.path.join(bin_dir, "julialauncher")
    with open(launcher, "w") as f:
        f.write("#!/bin/sh\n")
    os.chmod(launcher, os.stat(launcher).st_mode | stat.S_IEXEC)
    shim = os.path.join(bin_dir, "julia")
    os.symlink(launcher, shim)
    return shim


def _make_real_julia(bin_dir):
    """Create a fake plain (non-juliaup) julia binary — not a shim."""
    os.makedirs(bin_dir, exist_ok=True)
    exe = os.path.join(bin_dir, "julia")
    with open(exe, "w") as f:
        f.write("#!/bin/sh\n")
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
    return exe


class TestResolveJuliaupShim:
    def test_returns_none_when_julia_not_on_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, nothing on PATH
        assert _resolve_juliaup_shim() is None

    def test_returns_none_for_plain_non_juliaup_julia(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bin"
        _make_real_julia(str(bin_dir))
        monkeypatch.setenv("PATH", str(bin_dir))
        assert _resolve_juliaup_shim() is None

    def test_resolves_real_binary_behind_juliaup_shim(self, monkeypatch, tmp_path):
        shim_bin = tmp_path / "juliaup_bin"
        _make_shim(str(shim_bin))
        monkeypatch.setenv("PATH", str(shim_bin))

        # Fake ~/.julia/juliaup/ layout via JULIAUP_DEPOT_PATH override.
        depot = tmp_path / "depot"
        judir = depot / "juliaup"
        real_bin = judir / "julia-1.10.11+0.x64.linux.gnu" / "bin"
        real_exe = real_bin / "julia"
        os.makedirs(real_bin, exist_ok=True)
        with open(real_exe, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(real_exe, os.stat(real_exe).st_mode | stat.S_IEXEC)

        meta = {
            "Default": "release",
            "InstalledVersions": {
                "1.10.11+0.x64.linux.gnu": {"Path": "./julia-1.10.11+0.x64.linux.gnu"}
            },
            "InstalledChannels": {"release": {"Version": "1.10.11+0.x64.linux.gnu"}},
        }
        os.makedirs(judir, exist_ok=True)
        with open(judir / "juliaup.json", "w") as f:
            json.dump(meta, f)

        monkeypatch.setenv("JULIAUP_DEPOT_PATH", str(depot))
        resolved = _resolve_juliaup_shim()
        assert resolved == str(real_exe.resolve())

    def test_returns_none_when_juliaup_json_missing(self, monkeypatch, tmp_path):
        shim_bin = tmp_path / "juliaup_bin"
        _make_shim(str(shim_bin))
        monkeypatch.setenv("PATH", str(shim_bin))
        monkeypatch.setenv("JULIAUP_DEPOT_PATH", str(tmp_path / "nonexistent_depot"))
        assert _resolve_juliaup_shim() is None

    def test_returns_none_when_resolved_binary_does_not_exist(
        self, monkeypatch, tmp_path
    ):
        shim_bin = tmp_path / "juliaup_bin"
        _make_shim(str(shim_bin))
        monkeypatch.setenv("PATH", str(shim_bin))

        depot = tmp_path / "depot"
        judir = depot / "juliaup"
        os.makedirs(judir, exist_ok=True)
        meta = {
            "Default": "release",
            "InstalledVersions": {
                "1.10.11+0.x64.linux.gnu": {"Path": "./julia-1.10.11+0.x64.linux.gnu"}
            },
            "InstalledChannels": {"release": {"Version": "1.10.11+0.x64.linux.gnu"}},
        }
        with open(judir / "juliaup.json", "w") as f:
            json.dump(meta, f)
        # Note: no actual julia-1.10.11+.../bin/julia file created on disk.

        monkeypatch.setenv("JULIAUP_DEPOT_PATH", str(depot))
        assert _resolve_juliaup_shim() is None
