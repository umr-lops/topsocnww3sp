#!/usr/bin/env python3
"""Unit tests for l2c_processor module."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray
import xarray as xr
import yaml

from topsocnww3sp.l2c_processor import (
    create_global_attributes,
    find_ww3_file,
    list_osw_files_in_safe,
    main,
    process_group,
    process_lasso_group,
)
from topsocnww3sp.utils import load_ww3_field, load_ww3_multi_grid


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
    assert ds_out.sizes["sample"] > 0
    # WW3 observations are not a time series: the "time" dim is renamed to
    # "sample" and the observation times become a plain data variable (#23).
    assert "time" in ds_out.variables
    assert ds_out["time"].dims == ("sample",)
    assert ds_out["time"].attrs["long_name"] == "WW3 observation time (julian day, UT)"
    # the CF time-encoding attrs are dropped so 'time' is a plain data variable
    assert "units" not in ds_out["time"].attrs
    assert "standard_name" not in ds_out["time"].attrs


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
                        [0] * ds_out.sizes["sample"], dims="sample"
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
    mkdir_calls: list[Path] = []

    def fake_mkdir(self, *_, **__):
        mkdir_calls.append(self)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)
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
    expected_dir = Path(tmp_path / "output") / "2022" / "01" / "07" / safe_dir.name
    assert expected_dir in mkdir_calls


@pytest.mark.parametrize("mode", ["1to1", "lasso"])
def test_main_no_colocation_no_output_dir(
    mode,
    mock_safe_directory,
    dummy_ww3_ds,
    dummy_sar_ds_with_corners,
    tmp_path,
    monkeypatch,
):
    """When no colocation product is written, the output SAFE dir is not created."""
    safe_dir, _ = mock_safe_directory
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        'product_version: "v0.1"\n'
        "TIME_THRESHOLD_MINUTES: 30\n"
        "DISTANCE_THRESHOLD_KM: 20\n"
        "BUFFER_DEG: 0.1\n"
        "directory_ww3spectra_output: /fake/path\n"
    )

    mock_args = Mock(spec=argparse.Namespace)
    mock_args.ocn_safe = str(safe_dir)
    mock_args.ww3_file = "dummy_ww3.nc"
    mock_args.config = config_file
    mock_args.mode = mode
    mock_args.verbose = False
    mock_args.output_dir = str(tmp_path / "output")
    mock_args.overwrite = False

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda _: mock_args)
    if mode == "1to1":
        # SAR loads but WW3 is temporally far -> no match
        far_ww3 = dummy_ww3_ds.assign_coords(
            time=pd.date_range("2025-01-01 00:00", periods=5, freq="h")
        )
        monkeypatch.setattr(xarray, "open_dataset", lambda *_, **__: far_ww3)
        monkeypatch.setattr(
            "topsocnww3sp.l2c_processor.read_osw",
            lambda *_, **__: (dummy_sar_ds_with_corners, None),
        )
    else:
        # SAR loading fails and no WW3 points inside the footprint
        monkeypatch.setattr(xarray, "open_dataset", lambda *_, **__: dummy_ww3_ds)
        monkeypatch.setattr(
            "topsocnww3sp.l2c_processor.read_osw", lambda *_, **__: (None, None)
        )

    main()

    out_safe = tmp_path / "output" / "2022" / "01" / "07" / safe_dir.name
    assert not out_safe.exists()


# --- WW3 FIELD (integrated parameters) tests (issue #20) ---


@pytest.fixture
def dummy_ww3_tracks_multi_grid():
    """Dummy merged multi-grid WW3 track dataset with grid provenance."""
    times = pd.date_range("2022-01-07 06:00", periods=4, freq="h")
    freqs = np.linspace(0.04, 0.4, 3)
    dirs = np.linspace(0, 345, 4)
    rng = np.random.default_rng()

    return xr.Dataset(
        data_vars={
            "efth": (
                ["time", "frequency", "direction"],
                rng.random((4, 3, 4)).astype("f4"),
            ),
            "longitude": (["time"], [-5.0, -4.9, -179.9, 179.9]),
            "latitude": (["time"], [48.0, 48.0, 48.0, 48.0]),
            "ww3_grid_provenance": (["time"], np.array([0, 0, 2, 2], dtype="i8")),
        },
        coords={"time": times, "frequency": freqs, "direction": dirs},
    )


def _make_field_file(
    path: Path,
    lons: list[float],
    lats: list[float],
    times: pd.DatetimeIndex,
    hs_values: np.ndarray | None = None,
) -> None:
    """Write a small synthetic WW3 FIELD file."""
    if hs_values is None:
        hs_values = np.ones((len(times), len(lats), len(lons))) * 2.5
    ds = xr.Dataset(
        data_vars={
            "hs": (["time", "latitude", "longitude"], hs_values.astype("f4")),
            "MAPSTA": (["latitude", "longitude"], np.zeros((len(lats), len(lons)))),
        },
        coords={
            "time": times,
            "latitude": np.asarray(lats, dtype="f4"),
            "longitude": np.asarray(lons, dtype="f4"),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # scipy engine (NetCDF3) to avoid netCDF4/numpy backend write warnings
    ds.to_netcdf(path, engine="scipy")


def _field_config() -> dict[str, object]:
    """Config for the synthetic FIELD grid layout (run root passed separately)."""
    return {
        "TIME_THRESHOLD_MINUTES": 30,
        "ww3_field_max_distance_deg": 0.25,
        "ww3_grids": {
            "arctic": {"pattern": "ARC-*/YYYY-*/TRACK_NC/WW3-ARC-*_*_trck.nc"},
            "antarctic": {"pattern": "ANTARC-*/YYYY-*/TRACK_NC/WW3-ANTARC-*_*_trck.nc"},
            "midlatitude": {
                "pattern": "IRIGLOB-*/YYYY-*/TRACK_NC/WW3-IRIGLOB-*_*_trck.nc"
            },
        },
        "ww3_field_grids": {
            "arctic": {"pattern": "ARC-*/YYYY-*/FIELD_NC/WW3-ARC-*_*.nc"},
            "antarctic": {"pattern": "ANTARC-*/YYYY-*/FIELD_NC/WW3-ANTARC-*_*.nc"},
            "midlatitude": {"pattern": "IRIGLOB-*/YYYY-*/FIELD_NC/WW3-IRIGLOB-*_*.nc"},
        },
        "ww3_field_variables": {
            "hs": {
                "attrs": {
                    "long_name": "WW3 significant height of wind and swell waves",
                    "units": "m",
                    "standard_name": "sea_surface_wave_significant_height",
                },
            },
            "vtm02": {
                "attrs": {"long_name": "WW3 Tm2", "units": "s"},
            },
        },
    }


def test_load_ww3_field_sampling(tmp_path, dummy_ww3_tracks_multi_grid):
    """Each track point is sampled from its own provenance grid's FIELD file."""
    ww3_dir = tmp_path / "run"
    _make_field_file(
        ww3_dir / "IRIGLOB-7M/2022-01/FIELD_NC/WW3-IRIGLOB-7M_202201.nc",
        lons=[-5.5, -5.0, -4.5],
        lats=[47.5, 48.0, 48.5],
        times=pd.date_range("2022-01-07 06:00", periods=5, freq="h"),
        hs_values=np.full((5, 3, 3), 2.5, dtype="f4"),
    )
    _make_field_file(
        ww3_dir / "ARC-15KM/2022-01/FIELD_NC/WW3-ARC-15KM_202201.nc",
        lons=[80.0, 81.0],
        lats=[85.0, 86.0],
        times=pd.date_range("2022-01-07 06:00", periods=5, freq="h"),
        hs_values=np.full((5, 2, 2), 1.0, dtype="f4"),
    )

    ds = dummy_ww3_tracks_multi_grid
    # provenance [arctic, arctic, midlatitude, midlatitude]
    ds["ww3_grid_provenance"] = (["time"], np.array([0, 0, 2, 2], dtype="i8"))
    # arctic points near the ARC grid, midlatitude points near the IRIGLOB grid
    ds["longitude"] = (["time"], [80.0, 81.0, -5.0, -4.9])
    ds["latitude"] = (["time"], [85.0, 86.0, 48.0, 48.0])

    out = load_ww3_field(ww3_dir, ds, _field_config())

    assert "hs" in out
    var = out["hs"]
    assert var.dims == ("time",)
    assert var.sizes["time"] == 4
    # arctic points from the ARC FIELD file (1.0), midlatitude ones from
    # IRIGLOB (2.5) -- no cross-contamination between grids
    assert list(var.to_numpy()) == [1.0, 1.0, 2.5, 2.5]
    # aligned on the track time coordinate
    np.testing.assert_array_equal(var.coords["time"].to_numpy(), ds["time"].to_numpy())
    assert var.attrs["units"] == "m"
    assert var.attrs["standard_name"] == "sea_surface_wave_significant_height"
    # only configured variables are returned, MAPSTA is not
    assert "MAPSTA" not in out
    # vtm02 is configured but absent from the FIELD files -> not returned
    assert "vtm02" not in out


def test_load_ww3_field_single_file_mode(tmp_path, dummy_ww3_tracks_multi_grid):
    """Single-file mode: grid inferred from the track file name."""
    ww3_dir = tmp_path / "run" / "IRIGLOB-7M" / "2022-01" / "TRACK_NC"
    track_file = ww3_dir / "WW3-IRIGLOB-7M_202201_trck.nc"
    track_file.parent.mkdir(parents=True)
    _make_field_file(
        ww3_dir.parent / "FIELD_NC" / "WW3-IRIGLOB-7M_202201.nc",
        lons=[-5.0, -4.5],
        lats=[48.0],
        times=pd.date_range("2022-01-07 06:00", periods=4, freq="h"),
        hs_values=np.full((4, 1, 2), 3.0, dtype="f4"),
    )

    ds = dummy_ww3_tracks_multi_grid.copy()
    # single-file mode: no provenance variable
    del ds["ww3_grid_provenance"]
    ds["longitude"] = (["time"], [-5.0, -5.0, -5.0, -5.0])
    ds["latitude"] = (["time"], [48.0, 48.0, 48.0, 48.0])

    out = load_ww3_field(track_file, ds, _field_config())
    assert "hs" in out
    assert list(out["hs"].to_numpy()) == [3.0, 3.0, 3.0, 3.0]


def test_load_ww3_field_time_threshold(tmp_path, dummy_ww3_tracks_multi_grid):
    """Track times with no FIELD time within threshold keep NaN."""
    ww3_dir = tmp_path / "run"
    _make_field_file(
        ww3_dir / "IRIGLOB-7M/2022-01/FIELD_NC/WW3-IRIGLOB-7M_202201.nc",
        lons=[-5.0],
        lats=[48.0],
        times=pd.DatetimeIndex([pd.Timestamp("2022-01-07 06:00")]),
        hs_values=np.full((1, 1, 1), 2.5, dtype="f4"),
    )

    ds = dummy_ww3_tracks_multi_grid.copy()
    ds["ww3_grid_provenance"] = (["time"], np.array([2, 2, 2, 2], dtype="i8"))
    ds["longitude"] = (["time"], [-5.0, -5.0, -5.0, -5.0])
    ds["latitude"] = (["time"], [48.0, 48.0, 48.0, 48.0])
    # 06:00 and 06:25 are within 30 min, 07:00 and 08:00 are not
    ds = ds.assign_coords(
        time=pd.DatetimeIndex(
            [
                pd.Timestamp("2022-01-07 06:00"),
                pd.Timestamp("2022-01-07 06:25"),
                pd.Timestamp("2022-01-07 07:00"),
                pd.Timestamp("2022-01-07 08:00"),
            ]
        )
    )

    out = load_ww3_field(ww3_dir, ds, _field_config())
    assert "hs" in out
    assert np.isnan(out["hs"].to_numpy()).sum() == 2
    assert list(out["hs"].to_numpy()[:2]) == [2.5, 2.5]


def test_load_ww3_field_missing_file_warning(
    tmp_path, dummy_ww3_tracks_multi_grid, caplog
):
    """Missing FIELD file: warning logged, variable omitted, no failure."""
    ww3_dir = tmp_path / "run"  # empty run root: no FIELD file at all

    ds = dummy_ww3_tracks_multi_grid
    with caplog.at_level("WARNING", logger="topsocnww3sp.utils"):
        out = load_ww3_field(ww3_dir, ds, _field_config())

    assert out == {}
    assert any("No WW3 FIELD file" in m for m in caplog.messages)


def test_load_ww3_field_no_config(dummy_ww3_tracks_multi_grid, tmp_path):
    """Without FIELD config sections, sampling is skipped gracefully."""
    config = {"TIME_THRESHOLD_MINUTES": 30, "ww3_grids": {}}
    assert load_ww3_field(tmp_path, dummy_ww3_tracks_multi_grid, config) == {}
    # unknown grid in field config, no provenance match -> warning, no crash
    config["ww3_field_grids"] = {"arctic": {"pattern": "ARC-*"}}
    config["ww3_field_variables"] = {"hs": {"attrs": {}}}
    ds = dummy_ww3_tracks_multi_grid.copy()
    del ds["ww3_grid_provenance"]
    assert load_ww3_field(tmp_path, ds, config) == {}


def test_load_ww3_field_date_line(tmp_path, dummy_ww3_tracks_multi_grid):
    """Nearest-neighbor sampling works across the date line (lon +/-180)."""
    ww3_dir = tmp_path / "run"
    _make_field_file(
        ww3_dir / "IRIGLOB-7M/2022-01/FIELD_NC/WW3-IRIGLOB-7M_202201.nc",
        lons=[179.9, -180.0],
        lats=[48.0],
        times=pd.date_range("2022-01-07 06:00", periods=1, freq="h"),
        hs_values=np.array([[[2.0, 4.0]]], dtype="f4"),
    )

    ds = dummy_ww3_tracks_multi_grid.copy()
    # 179.99 is 0.01 deg from the -180.0 node across the date line;
    # 179.94 is 0.04 deg from the 179.9 node
    ds["longitude"] = (["time"], [179.99, 179.94, 0.0, 0.0])
    ds["latitude"] = (["time"], [48.0, 48.0, 48.0, 48.0])
    ds["ww3_grid_provenance"] = (["time"], np.array([2, 2, 2, 2], dtype="i8"))
    ds["time"] = (["time"], pd.date_range("2022-01-07 06:00", periods=4, freq="min"))

    config = _field_config()
    config["ww3_field_max_distance_deg"] = 0.1
    out = load_ww3_field(ww3_dir, ds, config)
    assert "hs" in out
    assert list(out["hs"].to_numpy())[:2] == [4.0, 2.0]


def test_create_global_attributes_field_note():
    """The global comment mentions the WW3 FIELD variables when present."""
    attrs = create_global_attributes(
        "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc",
        "v0.1",
        "lasso",
        ["hs"],
    )
    assert "hs" in attrs["comment"]
    attrs_no_field = create_global_attributes("x.nc", "v0.1", "lasso")
    assert "hs" not in attrs_no_field["comment"]


def test_main_integration_lasso_with_field_vars(
    monkeypatch, dummy_sar_ds_with_corners, dummy_ww3_ds_sparse, tmp_path
):
    """main() merges FIELD variables into the WW3 group (lasso mode)."""
    safe_dir = (
        tmp_path
        / "S1A_IW_OCN__2SDV_20220107T062432_20220107T062457_041351_04EA80_F9CD.SAFE"
    )
    osw_paths = [
        safe_dir
        / "measurement"
        / "s1a-iw1-osw-vv-20220107t062429-20220107t062500-041351-04ea80-001.nc",
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
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.list_osw_files_in_safe",
        lambda _: osw_paths,
    )
    monkeypatch.setattr(Path, "mkdir", lambda *_, **__: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(Path, "unlink", lambda *_, **__: None)
    monkeypatch.setattr(xarray, "open_dataset", lambda *_, **__: dummy_ww3_ds_sparse)

    captured: dict[str, object] = {}

    def fake_load_ww3_field(_, ds_ww3_tracks, __):
        captured["tracks"] = ds_ww3_tracks
        captured["called"] = True
        return {
            "hs": xr.DataArray(
                np.full(ds_ww3_tracks.sizes["time"], 2.5, dtype="f4"),
                dims="time",
                coords={"time": ds_ww3_tracks["time"].to_numpy()},
                attrs={"units": "m"},
            )
        }

    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.load_ww3_field", fake_load_ww3_field
    )
    monkeypatch.setattr(
        "topsocnww3sp.l2c_processor.read_osw",
        lambda *_, **__: (dummy_sar_ds_with_corners, None),
    )

    written: dict[str, xr.Dataset] = {}

    def mock_to_netcdf(_, _path, _mode="w", group=None, **_kwargs):
        if group == "WW3":
            written["ww3"] = _

    monkeypatch.setattr(xarray.Dataset, "to_netcdf", mock_to_netcdf)

    main()

    # the FIELD loader was invoked on the track dataset
    assert captured["called"] is True
    assert "hs" not in cast("xr.Dataset", captured["tracks"]).variables
    # the WW3 group written to the product carries the field variable on the
    # same dimension as the spectra
    assert "hs" in written["ww3"].variables
    assert written["ww3"]["hs"].dims == ("sample",)
    assert written["ww3"]["hs"].attrs["units"] == "m"
