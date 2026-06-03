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

from topsocnww3sp.l2c_processor import (
    find_ww3_file,
    main,
    process_group,
    process_lasso_group,
)


@pytest.fixture
def mock_config():
    return {
        "directory_ww3spectra_output": "/fake/path/ww3",
        "TIME_THRESHOLD_MINUTES": 30,
        "DISTANCE_THRESHOLD_KM": 20,
        "BUFFER_DEG": 0.1,
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
def dummy_ww3_ds_sparse():
    """Creates a dummy WW3 dataset with more scattered points for lasso testing."""
    times = pd.date_range("2022-01-07 06:00", periods=10, freq="h")
    freqs = np.linspace(0.04, 0.4, 3)
    dirs = np.linspace(0, 345, 4)
    rng = np.random.default_rng()

    # Create points scattered around the SAR area
    lons = [-5.2, -5.1, -5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3]
    lats = [48.2, 48.1, 48.0, 47.9, 47.8, 47.7, 47.6, 47.5, 47.4, 47.3]

    return xr.Dataset(
        data_vars={
            "efth": (
                ["time", "frequency", "direction"],
                rng.random((10, 3, 4)).astype("f4"),
            ),
            "longitude": (["time"], lons),
            "latitude": (["time"], lats),
            "dpt": (["time"], [100.0] * 10),
            "wnd": (["time"], [10.0] * 10),
            "wnddir": (["time"], [180.0] * 10),
            "cur": (["time"], [0.1] * 10),
            "curdir": (["time"], [90.0] * 10),
        },
        coords={"time": times, "frequency": freqs, "direction": dirs},
    )


@pytest.fixture
def dummy_sar_ds_with_corners():
    """Creates a dummy SAR dataset with proper corner coordinates."""
    ra = np.arange(2)
    az = np.arange(3)
    sub = [0]
    rng = np.random.default_rng()

    # Create a small rectangle for the footprint
    lon_center = -5.0
    lat_center = 48.0

    # Create corners for each tile (az, ra, 4 corners)
    lon_corners = np.zeros((1, 2, 3, 4))
    lat_corners = np.zeros((1, 2, 3, 4))

    for i in range(2):  # oswRaSize
        for j in range(3):  # oswAzSize
            # Define corners as a small rectangle around center + offset
            offset_lon = (i - 0.5) * 0.2
            offset_lat = (j - 1) * 0.15
            lon_corners[0, i, j, :] = [
                lon_center + offset_lon - 0.1,
                lon_center + offset_lon + 0.1,
                lon_center + offset_lon + 0.1,
                lon_center + offset_lon - 0.1,
            ]
            lat_corners[0, i, j, :] = [
                lat_center + offset_lat - 0.1,
                lat_center + offset_lat - 0.1,
                lat_center + offset_lat + 0.1,
                lat_center + offset_lat + 0.1,
            ]

    ds = xr.Dataset(
        data_vars={
            "oswLon": (
                ["subswath", "oswRaSize", "oswAzSize"],
                np.full((1, 2, 3), lon_center),
            ),
            "oswLat": (
                ["subswath", "oswRaSize", "oswAzSize"],
                np.full((1, 2, 3), lat_center),
            ),
            "oswHs": (
                ["subswath", "oswRaSize", "oswAzSize"],
                rng.random((1, 2, 3)).astype("f4"),
            ),
            "oswLongitudeCorner": (
                ["subswath", "oswRaSize", "oswAzSize", "oswCellCorner"],
                lon_corners,
            ),
            "oswLatitudeCorner": (
                ["subswath", "oswRaSize", "oswAzSize", "oswCellCorner"],
                lat_corners,
            ),
        },
        coords={
            "subswath": sub,
            "oswRaSize": ra,
            "oswAzSize": az,
            "oswCellCorner": [0, 1, 2, 3],
        },
    )
    return ds.stack(tiles=["oswRaSize", "oswAzSize"])  # noqa: PD013


# --- Tests for existing modes ---


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
def test_process_group_modes(
    mode, dummy_sar_ds_with_corners, dummy_ww3_ds, mock_config
):
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_sar, ds_ww3, ds_match = process_group(
            "fake_osw.nc", dummy_ww3_ds, "intraburst", mock_config, sar_start, mode
        )

    assert ds_sar is not None
    assert ds_ww3 is not None
    assert ds_match is not None


def test_process_group_no_temporal_match(
    dummy_sar_ds_with_corners, dummy_ww3_ds, mock_config
):
    sar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_sar, _, _ = process_group(
            "fake.nc", dummy_ww3_ds, "intraburst", mock_config, sar_start, "1to1"
        )
    assert ds_sar is None


# --- Tests for lasso mode ---


def test_process_lasso_group_success(
    dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, mock_config
):
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_out = process_lasso_group(
            "fake_osw.nc", dummy_ww3_ds_sparse, "intraburst", mock_config, sar_start
        )

    assert ds_out is not None
    assert "longitude" in ds_out.variables
    assert "latitude" in ds_out.variables
    # Remplacer ds_out.dims["time"] par ds_out.sizes["time"]
    assert ds_out.sizes["time"] > 0


def test_process_lasso_group_no_temporal_match(
    dummy_sar_ds_with_corners, dummy_ww3_ds, mock_config
):
    """Test lasso mode when no WW3 data within time window."""
    sar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_out = process_lasso_group(
            "fake_osw.nc", dummy_ww3_ds, "intraburst", mock_config, sar_start
        )

    assert ds_out is None


def test_process_lasso_group_no_spatial_match(
    dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, mock_config
):
    """Test lasso mode when WW3 points are far from SAR footprint."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    far_ww3_ds = dummy_ww3_ds_sparse.copy(deep=True)
    far_ww3_ds["longitude"] = far_ww3_ds["longitude"] + 100

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_out = process_lasso_group(
            "fake_osw.nc", far_ww3_ds, "intraburst", mock_config, sar_start
        )

    assert ds_out is None


def test_process_lasso_group_no_sar_data(mock_config):
    """Test lasso mode when SAR data loading fails."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)
    dummy_ww3 = xr.Dataset()

    with patch("topsocnww3sp.l2c_processor.read_osw", return_value=(None, None)):
        ds_out = process_lasso_group(
            "fake_osw.nc", dummy_ww3, "intraburst", mock_config, sar_start
        )

    assert ds_out is None


# --- Main integration test (updated for lasso) ---


def test_main_integration_1to1(monkeypatch, mock_config, dummy_ww3_ds):
    """Test main entry point with 1to1 mode."""
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
    monkeypatch.setattr(Path, "open", lambda *_, **__: MagicMock())
    monkeypatch.setattr(yaml, "safe_load", lambda _: mock_config)
    monkeypatch.setattr(xr, "open_dataset", lambda *_, **__: dummy_ww3_ds)
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)

    with (
        patch("xarray.Dataset.to_netcdf") as mock_save,
        patch("topsocnww3sp.l2c_processor.process_group") as mock_proc,
    ):
        mock_proc.return_value = (xr.Dataset(), xr.Dataset(), xr.Dataset())
        main()
        assert mock_save.called


def test_main_integration_lasso(
    monkeypatch, mock_config, dummy_sar_ds_with_corners, dummy_ww3_ds_sparse
):
    """Test main entry point with lasso mode."""
    mock_args = MagicMock()
    mock_args.osw_file = (
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc"
    )
    mock_args.ww3_file = "dummy_ww3.nc"
    mock_args.config = Path("/fake/path/config.yml")
    mock_args.mode = "lasso"
    mock_args.group = "intraburst"
    mock_args.verbose = False
    mock_args.output_dir = "/fake/output"
    mock_args.overwrite = False

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    monkeypatch.setattr(Path, "open", lambda *_, **__: MagicMock())
    monkeypatch.setattr(yaml, "safe_load", lambda _: mock_config)
    monkeypatch.setattr(xr, "open_dataset", lambda *_, **__: dummy_ww3_ds_sparse)
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)

    with (
        patch("xarray.Dataset.to_netcdf") as mock_save,
        patch(
            "topsocnww3sp.l2c_processor.read_osw",
            return_value=(dummy_sar_ds_with_corners, None),
        ),
    ):
        main()
        # Lasso mode should call to_netcdf at least once
        assert mock_save.called
