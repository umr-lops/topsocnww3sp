#!/usr/bin/env python3
"""Unit tests for l2c_processor module."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from topsocnww3sp.l2c_processor import find_ww3_file, main, process_group


@pytest.fixture
def mock_config():
    return {
        "directory_ww3spectra_output": "/fake/path/ww3",
        "TIME_THRESHOLD_MINUTES": 30,
        "DISTANCE_THRESHOLD_KM": 20,
    }


@pytest.fixture
def dummy_ww3_ds():
    """Creates a small dummy WW3 dataset."""
    times = pd.date_range("2022-01-07 06:00", periods=5, freq="h")
    freqs = np.linspace(0.04, 0.4, 3)
    dirs = np.linspace(0, 345, 4)
    rng = np.random.default_rng()

    return xr.Dataset(
        data_vars={
            "efth": (
                ["time", "frequency", "direction"],
                rng.random((5, 3, 4)).astype("f4"),
            ),
            "longitude": (["time"], [-5.0, -4.9, -4.8, -4.7, -4.6]),
            "latitude": (["time"], [48.0, 48.0, 48.0, 48.0, 48.0]),
            "dpt": (["time"], [100.0] * 5),
            "wnd": (["time"], [10.0] * 5),
            "wnddir": (["time"], [180.0] * 5),
            "cur": (["time"], [0.1] * 5),
            "curdir": (["time"], [90.0] * 5),
        },
        coords={"time": times, "frequency": freqs, "direction": dirs},
    )


@pytest.fixture
def dummy_sar_ds():
    """Creates a dummy SAR dataset mimicking your read_osw output."""
    ra = np.arange(2)
    az = np.arange(3)
    sub = [0]
    rng = np.random.default_rng()
    ds = xr.Dataset(
        data_vars={
            "oswLon": (
                ["subswath", "oswRaSize", "oswAzSize"],
                np.full((1, 2, 3), -5.0),
            ),
            "oswLat": (
                ["subswath", "oswRaSize", "oswAzSize"],
                np.full((1, 2, 3), 48.01),
            ),
            "oswHs": (
                ["subswath", "oswRaSize", "oswAzSize"],
                rng.random((1, 2, 3)).astype("f4"),
            ),
        },
        coords={"subswath": sub, "oswRaSize": ra, "oswAzSize": az},
    )
    # PD013: stack is appropriate here; silence the warning
    return ds.stack(tiles=["oswRaSize", "oswAzSize"])  # noqa: PD013


# --- Tests ---


def test_find_ww3_file(mock_config, monkeypatch):
    """Test find_ww3_file with a mock returning a file."""

    def mock_rglob(*_):  # unused arguments
        return ["/fake/path/ww3/WW3_202201_trck.nc"]

    monkeypatch.setattr(Path, "rglob", mock_rglob)

    sar_time = datetime(2022, 1, 7, 6, 25, tzinfo=timezone.utc)
    result = find_ww3_file(sar_time, mock_config)
    assert "202201" in result


def test_find_ww3_file_not_found(mock_config, monkeypatch):
    """Test find_ww3_file when no file is found (raises FileNotFoundError)."""

    def mock_rglob_empty(*_):
        return []

    monkeypatch.setattr(Path, "rglob", mock_rglob_empty)

    sar_time = datetime(2022, 1, 7, 6, 25, tzinfo=timezone.utc)
    with pytest.raises(FileNotFoundError):
        find_ww3_file(sar_time, mock_config)


@pytest.mark.parametrize("mode", ["1to1", "unique", "many"])
def test_process_group_modes(mode, dummy_sar_ds, dummy_ww3_ds, mock_config):
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw", return_value=(dummy_sar_ds, None)
    ):
        ds_sar, ds_ww3, ds_match = process_group(
            "fake_osw.nc", dummy_ww3_ds, "intraburst", mock_config, sar_start, mode
        )

    assert ds_sar is not None
    assert ds_ww3 is not None
    assert ds_match is not None


def test_process_group_no_temporal_match(dummy_sar_ds, dummy_ww3_ds, mock_config):
    sar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with patch(
        "topsocnww3sp.l2c_processor.read_osw", return_value=(dummy_sar_ds, None)
    ):
        ds_sar, _, _ = process_group(
            "fake.nc", dummy_ww3_ds, "intraburst", mock_config, sar_start, "1to1"
        )
    assert ds_sar is None


def test_main_integration(monkeypatch, mock_config, dummy_ww3_ds):
    """Test the main entry point logic without touching disk."""
    # 1. Mock CLI arguments
    mock_args = MagicMock()
    mock_args.osw_file = (
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc"
    )
    mock_args.ww3_file = "dummy_ww3.nc"
    mock_args.config = Path("/fake/path/config.yml")
    mock_args.mode = "1to1"
    mock_args.group = "intraburst"
    mock_args.verbose = False
    mock_args.output_dir = "/fake/output"
    mock_args.overwrite = False

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)

    # 2. Mock Path.open() to avoid real file opening
    monkeypatch.setattr(Path, "open", lambda *_, **__: MagicMock())

    # 3. Mock yaml.safe_load to return config
    monkeypatch.setattr(yaml, "safe_load", lambda _: mock_config)

    # 4. Mock xarray.open_dataset
    monkeypatch.setattr(xr, "open_dataset", lambda *_, **__: dummy_ww3_ds)

    # 5. Mock filesystem methods to avoid real access
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)

    # 6. Mock to_netcdf and process_group
    with (
        patch("xarray.Dataset.to_netcdf") as mock_save,
        patch("topsocnww3sp.l2c_processor.process_group") as mock_proc,
    ):
        mock_proc.return_value = (xr.Dataset(), xr.Dataset(), xr.Dataset())

        main()

        assert mock_save.called
