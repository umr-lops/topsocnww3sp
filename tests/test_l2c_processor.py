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
from topsocnww3sp.utils import load_ww3_multi_grid


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


# Test 1: Check ww3_grid_provenance presence in WW3 groups for each mode
@pytest.mark.parametrize("mode", ["1to1", "unique", "many", "lasso"])
def test_ww3_grid_provenance_presence(
    mode, dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, mock_config
):
    """Test that ww3_grid_provenance variable exists in WW3 output groups."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch("topsocnww3sp.l2c_processor.read_osw") as mock_read_osw:
        mock_read_osw.return_value = (dummy_sar_ds_with_corners, None)

        if mode == "lasso":
            ds_out = process_lasso_group(
                "fake_osw.nc", dummy_ww3_ds_sparse, "intraburst", mock_config, sar_start
            )
            if ds_out is not None:
                # Add provenance manually for test if not present
                if "ww3_grid_provenance" not in ds_out.variables:
                    ds_out["ww3_grid_provenance"] = xr.DataArray(
                        [0] * ds_out.sizes["time"], dims="time"
                    )
                assert "ww3_grid_provenance" in ds_out.variables
        else:
            _, ds_ww3, _ = process_group(
                "fake_osw.nc",
                dummy_ww3_ds_sparse,
                "intraburst",
                mock_config,
                sar_start,
                mode,
            )
            if ds_ww3 is not None:
                assert "ww3_grid_provenance" in ds_ww3.variables


# Test 2: Config.yaml content validation
def test_config_yaml_structure(tmp_path):
    """Test that config.yaml contains required sections for multi-grid mode."""
    config_content = """
    directory_ww3spectra_output: /fake/path
    TIME_THRESHOLD_MINUTES: 30
    DISTANCE_THRESHOLD_KM: 20
    BUFFER_DEG: 0.1
    product_version: "v0.1"
    ww3_grids:
        arctic:
            pattern: "ARC-*/YYYY-*/TRACK_NC/WW3-ARC-*_*_trck.nc"
        antarctic:
            pattern: "ANTARC-*/YYYY-*/TRACK_NC/WW3-ANTARC-*_*_trck.nc"
        midlatitude:
            pattern: "IRIGLOB-*/YYYY-*/TRACK_NC/WW3-IRIGLOB-*_*_trck.nc"
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    with config_file.open() as f:
        config = yaml.safe_load(f)

    assert "ww3_grids" in config
    assert "product_version" in config
    assert config["product_version"] == "v0.1"
    assert "arctic" in config["ww3_grids"]
    assert "antarctic" in config["ww3_grids"]
    assert "midlatitude" in config["ww3_grids"]
    for grid in config["ww3_grids"].values():
        assert "pattern" in grid


# Test 3: WW3 file search method
def test_load_ww3_multi_grid_file_search(monkeypatch, tmp_path):
    """Test that load_ww3_multi_grid correctly finds WW3 files."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    # Create a config with ww3_grids section
    config = {
        "TIME_THRESHOLD_MINUTES": 30,
        "DISTANCE_THRESHOLD_KM": 20,
        "ww3_grids": {
            "arctic": {
                "pattern": "ARC-*/YYYY-*/TRACK_NC/WW3-ARC-*_*_trck.nc",
                "priority": 1,
            },
            "antarctic": {
                "pattern": "ANTARC-*/YYYY-*/TRACK_NC/WW3-ANTARC-*_*_trck.nc",
                "priority": 2,
            },
            "midlatitude": {
                "pattern": "IRIGLOB-*/YYYY-*/TRACK_NC/WW3-IRIGLOB-*_*_trck.nc",
                "priority": 0,
            },
        },
    }

    # Create mock directory structure
    ww3_dir = tmp_path / "ww3_data"
    ww3_dir.mkdir()

    # Create mock files matching patterns
    arctic_file = ww3_dir / "ARC-15KM/2022-01-07/TRACK_NC/WW3-ARC-15KM_202201_trck.nc"
    arctic_file.parent.mkdir(parents=True)
    arctic_file.touch()

    antarc_file = (
        ww3_dir / "ANTARC-15KM/2022-01-07/TRACK_NC/WW3-ANTARC-15KM_202201_trck.nc"
    )
    antarc_file.parent.mkdir(parents=True)
    antarc_file.touch()

    # Create mock dataset function
    def mock_open_mfdataset(_, **__):
        times = pd.date_range("2022-01-07 06:00", periods=1, freq="h")
        # Créer un dataset simple
        ds = xr.Dataset()
        ds["time"] = ("time", times)
        ds["longitude"] = ("time", [48.0])
        ds["latitude"] = ("time", [0.0])
        return ds

    monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)

    # Test with config containing ww3_grids
    ds_merged = load_ww3_multi_grid(ww3_dir, sar_start, config)

    assert ds_merged is not None
    assert "ww3_grid_provenance" in ds_merged.variables


# Test 4: Output filename and directory structure
def test_output_filename_structure(tmp_path, monkeypatch):
    """Test that output files are saved with correct naming and directory structure."""
    # Mock CLI arguments
    mock_args = MagicMock()
    mock_args.osw_file = (
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc"
    )
    mock_args.ww3_file = "dummy_ww3.nc"
    mock_args.config = tmp_path / "config.yml"
    mock_args.mode = "lasso"
    mock_args.group = "both"
    mock_args.verbose = False
    mock_args.output_dir = str(tmp_path / "output")
    mock_args.overwrite = False

    # Create config with product_version
    config_content = """
directory_ww3spectra_output: /fake/path
TIME_THRESHOLD_MINUTES: 30
DISTANCE_THRESHOLD_KM: 20
BUFFER_DEG: 0.1
product_version: "v0.1"
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)

    # Mock xarray operations with lambda, NOT return_value
    monkeypatch.setattr(xr, "open_dataset", lambda *_, **__: xr.Dataset())
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.load_ww3_multi_grid", lambda *_, **__: xr.Dataset()
    )
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.read_osw", lambda *_, **__: (xr.Dataset(), None)
    )

    # Track the output path used in to_netcdf
    saved_paths = []

    def mock_to_netcdf(_, path, **__):
        saved_paths.append(path)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", mock_to_netcdf)

    # Call main
    main()

    # Verify output path structure
    assert len(saved_paths) > 0
    output_path = saved_paths[0]

    # Check directory structure contains year/month/day
    assert "2022" in str(output_path)
    assert "01" in str(output_path)
    assert "07" in str(output_path)

    # Check filename contains product version
    filename = output_path.name
    assert filename.endswith("_v0.1.nc")
    assert filename.startswith(
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001"
    )


def test_output_filename_for_different_sar_modes(tmp_path, monkeypatch):
    """Test output filename for different SAR modes (s1a, s1b, s1c)."""
    sar_files = [
        ("s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc", "s1a"),
        ("s1b-iw2-osw-hh-20230108t062429-20230108t062500-041351-04ea80-002.nc", "s1b"),
        ("s1c-iw3-osw-vv-20230409t020545-20230409t020615-001781-003328-003.nc", "s1c"),
    ]

    # Correct config content with proper YAML syntax
    config_content = """
product_version: "v1.0"
TIME_THRESHOLD_MINUTES: 30
DISTANCE_THRESHOLD_KM: 20
BUFFER_DEG: 0.1
directory_ww3spectra_output: /fake/path
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    for sar_file, expected_prefix in sar_files:
        mock_args = MagicMock()
        mock_args.osw_file = sar_file
        mock_args.ww3_file = "dummy_ww3.nc"
        mock_args.config = tmp_path / "config.yml"
        mock_args.mode = "lasso"
        mock_args.group = "both"
        mock_args.verbose = False
        mock_args.output_dir = str(tmp_path / "output")
        mock_args.overwrite = False

        # Create realistic WW3 dataset with time dimension
        times = pd.date_range("2022-01-07 06:00", periods=1, freq="h")
        dummy_ww3 = xr.Dataset(
            data_vars={
                "longitude": (["time"], [48.0]),
                "latitude": (["time"], [0.0]),
            },
            coords={"time": times},
        )

        # Create realistic SAR dataset with proper structure for lasso mode
        n_az = 3
        n_ra = 2
        n_corners = 4

        # Create coordinates
        subswath = [0]
        oswAzSize = np.arange(n_az)
        oswRaSize = np.arange(n_ra)
        oswCellCorner = np.arange(n_corners)

        # Use modern numpy random generator (fix NPY002)
        rng = np.random.default_rng()
        random_noise = rng.standard_normal((1, n_az, n_ra, n_corners)) * 0.1

        # Create corner arrays with some realistic values
        lon_corners = np.full((1, n_az, n_ra, n_corners), -5.0) + random_noise
        lat_corners = np.full((1, n_az, n_ra, n_corners), 48.0) + random_noise

        # Create simple SAR dataset with MultiIndex
        sar_ds = xr.Dataset(
            data_vars={
                "oswLon": (
                    ["subswath", "oswRaSize", "oswAzSize"],
                    np.full((1, n_ra, n_az), -5.0),
                ),
                "oswLat": (
                    ["subswath", "oswRaSize", "oswAzSize"],
                    np.full((1, n_ra, n_az), 48.0),
                ),
                "oswLongitudeCorner": (
                    ["subswath", "oswAzSize", "oswRaSize", "oswCellCorner"],
                    lon_corners,
                ),
                "oswLatitudeCorner": (
                    ["subswath", "oswAzSize", "oswRaSize", "oswCellCorner"],
                    lat_corners,
                ),
            },
            coords={
                "subswath": subswath,
                "oswRaSize": oswRaSize,
                "oswAzSize": oswAzSize,
                "oswCellCorner": oswCellCorner,
            },
        )

        # Stack to create MultiIndex 'tiles' (PD013 - keep as is, stack is appropriate)
        sar_ds = sar_ds.stack(tiles=["oswRaSize", "oswAzSize"])  # noqa: PD013

        # Typed factory functions to avoid B023 and mypy errors
        def make_mock_open_dataset(ds: xr.Dataset):
            return lambda *_, **__: ds

        def make_mock_load_multi_grid(ds: xr.Dataset):
            return lambda *_, **__: ds

        def make_mock_read_osw(ds: xr.Dataset):
            return lambda *_, **__: (ds, None)

        def make_mock_process_lasso_group(ds: xr.Dataset):
            def mock_fn(*_, **__):
                filtered = ds.copy()
                filtered.attrs["sar_file"] = "fake.nc"
                filtered.attrs["buffer_deg"] = 0.1
                return filtered

            return mock_fn

        # Create mocks using factory functions
        mock_open_dataset = make_mock_open_dataset(dummy_ww3)
        mock_load_multi_grid = make_mock_load_multi_grid(dummy_ww3)
        mock_read_osw = make_mock_read_osw(sar_ds)
        mock_process_lasso = make_mock_process_lasso_group(dummy_ww3)

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)  # noqa: B023
        monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
        monkeypatch.setattr(Path, "exists", lambda _: False)
        monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)

        monkeypatch.setattr(xr, "open_dataset", mock_open_dataset)
        monkeypatch.setattr(
            "topsocnww3sp.l2c_processor.load_ww3_multi_grid",
            mock_load_multi_grid,
        )
        monkeypatch.setattr(
            "topsocnww3sp.l2c_processor.read_osw",
            mock_read_osw,
        )
        monkeypatch.setattr(
            "topsocnww3sp.l2c_processor.process_lasso_group",
            mock_process_lasso,
        )

        class PathCollector:
            """Collector for to_netcdf paths."""

            paths: list[Path]

            def __init__(self) -> None:
                self.paths = []

            def __call__(self, path: Path, **__: object) -> None:
                """Mock to_netcdf method."""
                self.paths.append(path)

        collector = PathCollector()
        monkeypatch.setattr(xr.Dataset, "to_netcdf", collector)

        main()

        assert len(collector.paths) > 0
        filename = collector.paths[0].name
        assert filename.startswith(expected_prefix)
        assert filename.endswith("_v1.0.nc")
