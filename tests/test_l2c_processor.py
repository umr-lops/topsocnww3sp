import argparse
import glob
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

# Import functions from your script
from topsocnww3sp.l2c_processor import find_ww3_file, haversine, main, process_group


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
    # FIXED: changed 'H' to 'h' for compatibility with newer Pandas
    times = pd.date_range("2022-01-07 06:00", periods=5, freq="h")
    freqs = np.linspace(0.04, 0.4, 3)
    dirs = np.linspace(0, 345, 4)

    ds = xr.Dataset(
        data_vars={
            "efth": (
                ["time", "frequency", "direction"],
                np.random.rand(5, 3, 4).astype("f4"),
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
    return ds


@pytest.fixture
def dummy_sar_ds():
    """Creates a dummy SAR dataset mimicking your read_osw output."""
    ra = np.arange(2)
    az = np.arange(3)
    sub = [0]

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
            "oswHs": (["subswath", "oswRaSize", "oswAzSize"], np.random.rand(1, 2, 3)),
        },
        coords={"subswath": sub, "oswRaSize": ra, "oswAzSize": az},
    )
    return ds.stack(tiles=["oswRaSize", "oswAzSize"])


# --- Tests ---


def test_haversine():
    dist = haversine(0, 0, 1, 0)
    assert dist == pytest.approx(111.19, rel=1e-3)


def test_find_ww3_file(mock_config, monkeypatch):
    # FIXED: using built-in monkeypatch instead of mocker
    def mock_glob(pattern, recursive=False):
        return ["/fake/path/ww3/WW3_202201_trck.nc"]

    monkeypatch.setattr(glob, "glob", mock_glob)

    sar_time = datetime(2022, 1, 7, 6, 25)
    result = find_ww3_file(sar_time, mock_config)
    assert "202201" in result


def test_find_ww3_file_not_found(mock_config, monkeypatch):
    monkeypatch.setattr("glob.glob", lambda x, recursive=False: [])
    with pytest.raises(FileNotFoundError):
        find_ww3_file(datetime(2022, 1, 7), mock_config)


@pytest.mark.parametrize("mode", ["1to1", "unique", "many"])
def test_process_group_modes(mode, dummy_sar_ds, dummy_ww3_ds, mock_config):
    sar_start = datetime(2022, 1, 7, 6, 0)

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
    sar_start = datetime(2025, 1, 1)
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
    mock_args.config = "config.yml"
    mock_args.mode = "1to1"
    mock_args.group = "intraburst"

    # Fix ARG005 by replacing 'self' with '_'
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)

    # Also fix 'x' here if the linter flags it
    monkeypatch.setattr(yaml, "safe_load", lambda _: mock_config)

    monkeypatch.setattr("builtins.open", MagicMock())

    # 3. Mock Xarray and logic
    monkeypatch.setattr(xr, "open_dataset", lambda *_args, **_kwargs: dummy_ww3_ds)

    with patch("xarray.Dataset.to_netcdf") as mock_save:
        with patch("topsocnww3sp.l2c_processor.process_group") as mock_proc:
            mock_proc.return_value = (xr.Dataset(), xr.Dataset(), xr.Dataset())

            main()

            assert mock_save.called
