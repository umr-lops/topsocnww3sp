#!/usr/bin/env python3
"""Unit tests for read_s1_osw_tops_data.read_osw."""

from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest
import xarray as xr

from topsocnww3sp.read_s1_osw_tops_data import read_osw


def _write_osw_file(path: Path, lon: float = -5.0, land_flag: int = 0) -> None:
    """Write a minimal OSW .nc file with an intraburst group."""
    rng = np.random.default_rng()
    ds = xr.Dataset(
        data_vars={
            "oswLon": (["oswRaSize", "oswAzSize"], np.full((2, 3), lon)),
            "oswLat": (["oswRaSize", "oswAzSize"], np.full((2, 3), 48.0)),
            "oswHs": (
                ["oswRaSize", "oswAzSize"],
                rng.random((2, 3)).astype("f4"),
            ),
            "oswLandFlag": (
                ["oswRaSize", "oswAzSize"],
                np.full((2, 3), land_flag, "i1"),
            ),
            "oswCoh": (
                ["oswRaSize", "oswAzSize", "oswKyBinSize"],
                np.ones((2, 3, 50), "f4"),
            ),
        },
        coords={
            "oswRaSize": np.arange(2, dtype="i8"),
            "oswAzSize": np.arange(3, dtype="i8"),
            "oswKyBinSize": np.arange(50, dtype="i8"),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Groups require the netCDF4 engine (NetCDF3 has no groups). With the
    # netCDF4/numpy backend a benign DeprecationWarning can be emitted while
    # writing, so the suite-wide filterwarnings=error is relaxed just for this
    # call.
    with catch_warnings():
        simplefilter("ignore", DeprecationWarning)
        ds.to_netcdf(path, group="intraburst")


def test_read_osw_single_file_no_subswath_dim(tmp_path):
    """A single OSW file must not produce a size-1 subswath dimension (issue #22)."""
    osw_path = (
        tmp_path / "measurement" / "s1a-iw1-osw-vv-20220107t062429-041351-04ea80-001.nc"
    )
    _write_osw_file(osw_path)

    fat_osw, coords_osw = read_osw("intraburst", [osw_path])

    assert "subswath" not in fat_osw.dims
    assert "subswath" not in fat_osw.coords
    assert fat_osw.sizes["tiles"] == 6
    assert len(coords_osw["lon_osw"]) == 6
    assert len(coords_osw["lat_osw"]) == 6


def test_read_osw_all_land_returns_empty(tmp_path):
    """When every file is land-only, read_osw raises on the empty concat."""
    osw_path = (
        tmp_path / "measurement" / "s1a-iw2-osw-vv-20220107t062430-041351-04ea80-002.nc"
    )
    _write_osw_file(osw_path, land_flag=1)

    with pytest.raises(ValueError, match="at least one object"):
        read_osw("intraburst", [osw_path])
