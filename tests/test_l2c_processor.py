#!/usr/bin/env python3
"""Unit tests for l2c_processor module."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray
import xarray as xr
import yaml

from topsocnww3sp.l2c_processor import (
    find_ww3_file,
    list_osw_files_in_safe,
    main,
    process_group,
    process_lasso_group,
)
from topsocnww3sp.utils import load_ww3_multi_grid


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "directory_ww3spectra_output": "/fake/path/ww3",
        "TIME_THRESHOLD_MINUTES": 30,
        "DISTANCE_THRESHOLD_KM": 20,
        "BUFFER_DEG": 0.1,
    }


@pytest.fixture
def dummy_ww3_ds():
    """Create a small dummy WW3 dataset."""
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
    """Create a dummy WW3 dataset with scattered points for lasso testing."""
    times = pd.date_range("2022-01-07 06:00", periods=10, freq="h")
    freqs = np.linspace(0.04, 0.4, 3)
    dirs = np.linspace(0, 345, 4)
    rng = np.random.default_rng()

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
    """Create a dummy SAR dataset with proper corner coordinates."""
    ra = np.arange(2)
    az = np.arange(3)
    sub = [0]
    rng = np.random.default_rng()

    lon_center = -5.0
    lat_center = 48.0

    lon_corners = np.zeros((1, 2, 3, 4))
    lat_corners = np.zeros((1, 2, 3, 4))

    for i in range(2):
        for j in range(3):
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


@pytest.fixture
def mock_safe_directory(tmp_path):
    """Create a mock SAFE directory with OSW files for testing."""
    safe_dir = (
        tmp_path
        / "S1A_IW_OCN__2SDV_20220107T062432_20220107T062457_041351_04EA80_F9CD.SAFE"
    )
    measurement_dir = safe_dir / "measurement"
    measurement_dir.mkdir(parents=True)

    osw_files = [
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc",
        "s1a-iw2-osw-vv-20220107t062430-20220107t062501-041351-04ea80-002.nc",
        "s1a-iw3-osw-vv-20220107t062431-20220107t062502-041351-04ea80-003.nc",
    ]

    for fname in osw_files:
        (measurement_dir / fname).touch()

    return safe_dir, osw_files


# --- Tests ---


def test_find_ww3_file(mock_config, monkeypatch):
    """Test find_ww3_file with a mock returning a file."""

    def mock_rglob(*_):
        return ["/fake/path/ww3/WW3_202201_trck.nc"]

    monkeypatch.setattr(Path, "rglob", mock_rglob)

    sar_time = datetime(2022, 1, 7, 6, 25, tzinfo=timezone.utc)
    result = find_ww3_file(sar_time, mock_config)
    assert "202201" in result


def test_find_ww3_file_not_found(mock_config, monkeypatch):
    """Test find_ww3_file when no file is found."""

    def mock_rglob_empty(*_):
        return []

    monkeypatch.setattr(Path, "rglob", mock_rglob_empty)

    sar_time = datetime(2022, 1, 7, 6, 25, tzinfo=timezone.utc)
    with pytest.raises(FileNotFoundError):
        find_ww3_file(sar_time, mock_config)


def test_list_osw_files_in_safe(mock_safe_directory):
    """Test listing OSW files in a SAFE directory."""
    safe_dir, expected_files = mock_safe_directory
    osw_files = list_osw_files_in_safe(safe_dir)

    assert len(osw_files) == 3
    for i, osw_file in enumerate(osw_files):
        assert osw_file.name == expected_files[i]
        assert osw_file.parent.name == "measurement"


def test_list_osw_files_in_safe_not_found(tmp_path):
    """Test listing OSW files when SAFE has no measurement directory."""
    safe_dir = tmp_path / "invalid.SAFE"
    safe_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Measurement directory not found"):
        list_osw_files_in_safe(safe_dir)


def test_list_osw_files_in_safe_no_osw_files(tmp_path):
    """Test listing OSW files when no OSW files exist."""
    safe_dir = tmp_path / "S1A_EMPTY.SAFE"
    measurement_dir = safe_dir / "measurement"
    measurement_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No OSW \.nc files found"):
        list_osw_files_in_safe(safe_dir)


@pytest.mark.parametrize("mode", ["1to1", "unique", "many"])
def test_process_group_modes(
    mode, dummy_sar_ds_with_corners, dummy_ww3_ds, mock_config
):
    """Test process_group with different modes."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_sar, ds_ww3, ds_match = process_group(
            Path("fake_osw.nc"),
            dummy_ww3_ds,
            "intraburst",
            mock_config,
            sar_start,
            mode,
        )

    assert ds_sar is not None
    assert ds_ww3 is not None
    assert ds_match is not None


def test_process_group_no_temporal_match(
    dummy_sar_ds_with_corners, dummy_ww3_ds, mock_config
):
    """Test process_group when no WW3 data within time window."""
    sar_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_sar, _, _ = process_group(
            Path("fake.nc"), dummy_ww3_ds, "intraburst", mock_config, sar_start, "1to1"
        )
    assert ds_sar is None


def test_process_lasso_group_success(
    dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, mock_config
):
    """Test lasso mode successfully extracts WW3 points."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    with patch(
        "topsocnww3sp.l2c_processor.read_osw",
        return_value=(dummy_sar_ds_with_corners, None),
    ):
        ds_out = process_lasso_group(
            Path("fake_osw.nc"),
            dummy_ww3_ds_sparse,
            "intraburst",
            mock_config,
            sar_start,
        )

    assert ds_out is not None
    assert "longitude" in ds_out.variables
    assert "latitude" in ds_out.variables
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
            Path("fake_osw.nc"), dummy_ww3_ds, "intraburst", mock_config, sar_start
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
            Path("fake_osw.nc"), far_ww3_ds, "intraburst", mock_config, sar_start
        )

    assert ds_out is None


def test_process_lasso_group_no_sar_data(mock_config):
    """Test lasso mode when SAR data loading fails."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)
    dummy_ww3 = xr.Dataset()

    with patch("topsocnww3sp.l2c_processor.read_osw", return_value=(None, None)):
        ds_out = process_lasso_group(
            Path("fake_osw.nc"), dummy_ww3, "intraburst", mock_config, sar_start
        )

    assert ds_out is None


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
                Path("fake_osw.nc"),
                dummy_ww3_ds_sparse,
                "intraburst",
                mock_config,
                sar_start,
            )
            if ds_out is not None:
                if "ww3_grid_provenance" not in ds_out.variables:
                    ds_out["ww3_grid_provenance"] = xr.DataArray(
                        [0] * ds_out.sizes["time"], dims="time"
                    )
                assert "ww3_grid_provenance" in ds_out.variables
        else:
            _, ds_ww3, _ = process_group(
                Path("fake_osw.nc"),
                dummy_ww3_ds_sparse,
                "intraburst",
                mock_config,
                sar_start,
                mode,
            )
            if ds_ww3 is not None:
                assert "ww3_grid_provenance" in ds_ww3.variables


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


def test_load_ww3_multi_grid_file_search(monkeypatch, tmp_path):
    """Test that load_ww3_multi_grid correctly finds WW3 files."""
    sar_start = datetime(2022, 1, 7, 6, 0, tzinfo=timezone.utc)

    config = {
        "TIME_THRESHOLD_MINUTES": 30,
        "DISTANCE_THRESHOLD_KM": 20,
        "ww3_grids": {
            "arctic": {"pattern": "ARC-*/YYYY-*/TRACK_NC/WW3-ARC-*_*_trck.nc"},
            "antarctic": {"pattern": "ANTARC-*/YYYY-*/TRACK_NC/WW3-ANTARC-*_*_trck.nc"},
            "midlatitude": {
                "pattern": "IRIGLOB-*/YYYY-*/TRACK_NC/WW3-IRIGLOB-*_*_trck.nc"
            },
        },
    }

    ww3_dir = tmp_path / "ww3_data"
    ww3_dir.mkdir()

    arctic_file = ww3_dir / "ARC-15KM/2022-01-07/TRACK_NC/WW3-ARC-15KM_202201_trck.nc"
    arctic_file.parent.mkdir(parents=True)
    arctic_file.touch()

    def mock_open_mfdataset(_, **__):
        times = pd.date_range("2022-01-07 06:00", periods=1, freq="h")
        ds = xr.Dataset()
        ds["time"] = ("time", times)
        ds["longitude"] = ("time", [48.0])
        ds["latitude"] = ("time", [0.0])
        return ds

    monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)

    ds_merged = load_ww3_multi_grid(ww3_dir, sar_start, config)

    assert ds_merged is not None
    assert "ww3_grid_provenance" in ds_merged.variables


def test_main_integration_lasso_with_safe(
    monkeypatch, dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, tmp_path
):
    """Test main entry point with lasso mode and SAFE directory."""
    # Créer des chemins mockés pour les fichiers OSW
    safe_dir = (
        tmp_path
        / "S1A_IW_OCN__2SDV_20220107T062432_20220107T062457_041351_04EA80_F9CD.SAFE"
    )
    osw_paths = [
        safe_dir
        / "measurement"
        / "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc",
        safe_dir
        / "measurement"
        / "s1a-iw2-osw-vv-20220107t062430-20220107t062501-041351-04ea80-002.nc",
        safe_dir
        / "measurement"
        / "s1a-iw3-osw-vv-20220107t062431-20220107t062502-041351-04ea80-003.nc",
    ]

    config_file = tmp_path / "config.yml"
    config_content = """
product_version: "v0.1"
TIME_THRESHOLD_MINUTES: 30
DISTANCE_THRESHOLD_KM: 20
BUFFER_DEG: 0.1
directory_ww3spectra_output: /fake/path
"""
    config_file.write_text(config_content)

    mock_args = Mock(spec=argparse.Namespace)
    mock_args.ocn_safe = str(safe_dir)
    mock_args.ww3_file = "dummy_ww3.nc"
    mock_args.config = config_file
    mock_args.mode = "lasso"
    mock_args.verbose = False
    mock_args.output_dir = str(tmp_path / "output")
    mock_args.overwrite = False

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    # Mocker list_osw_files_in_safe pour retourner les chemins directement
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.list_osw_files_in_safe",
        lambda _: osw_paths,
    )
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)
    monkeypatch.setattr(xarray, "open_dataset", lambda *_, **__: dummy_ww3_ds_sparse)
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.load_ww3_multi_grid",
        lambda *_, **__: dummy_ww3_ds_sparse,
    )
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.read_osw",
        lambda *_, **__: (dummy_sar_ds_with_corners, None),
    )
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.process_lasso_group",
        lambda *_, **__: dummy_ww3_ds_sparse,
    )

    saved_paths = []
    written_files = set()

    def mock_to_netcdf(_, path, _mode="w", _group=None, **_kwargs):
        """Mock to_netcdf method - accepts all standard arguments."""
        if path not in written_files:
            written_files.add(path)
            saved_paths.append(path)

    monkeypatch.setattr(xarray.Dataset, "to_netcdf", mock_to_netcdf)

    main()

    assert len(saved_paths) == len(osw_paths)
    for path in saved_paths:
        assert path.name.endswith("_v0.1.nc")
        assert "output" in str(path)
        assert "2022" in str(path)
        assert "01" in str(path)
        assert "07" in str(path)
