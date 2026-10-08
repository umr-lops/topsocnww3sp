#!/usr/bin/env python3
"""Unit tests for diagnostic_map_progressive (offline: network + file IO mocked)."""

import sys
from pathlib import Path
from unittest.mock import patch

import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
from shapely.geometry import Polygon

import topsocnww3sp.diagnostic_map_progressive as dmp


@pytest.fixture(autouse=True)
def _use_agg():
    """Force the non-interactive Agg backend so no display is required."""
    mpl.use("Agg")


def _sar_ds() -> xr.Dataset:
    """A tiny SAR dataset with one 1x1-degree tile and three centre points."""
    lon_corners = np.array([[-5.5, -4.5, -4.5, -5.5]])
    lat_corners = np.array([[47.5, 47.5, 48.5, 48.5]])
    return xr.Dataset(
        data_vars={
            "oswLon": ("tiles", [-5.1, -5.0, -4.9]),
            "oswLat": ("tiles", [47.9, 48.1, 47.9]),
            "oswLongitudeCorner": (("oswTile", "oswCellCorner"), lon_corners),
            "oswLatitudeCorner": (("oswTile", "oswCellCorner"), lat_corners),
        }
    )


def _sar_ds_no_corners() -> xr.Dataset:
    """A SAR dataset that has centre points but no corner variables."""
    return xr.Dataset(
        data_vars={
            "oswLon": ("tiles", [-5.0]),
            "oswLat": ("tiles", [48.0]),
        }
    )


def _ww3_ds() -> xr.Dataset:
    """A tiny WW3 dataset with 1-D longitude/latitude."""
    return xr.Dataset(
        data_vars={
            "longitude": ("time", [-5.0, -4.9, -4.8]),
            "latitude": ("time", [48.0, 48.0, 48.1]),
        }
    )


@pytest.fixture
def geo_axes():
    """A real cartopy GeoAxes for plotting tests (figure closed on teardown)."""
    fig = mpl.pyplot.figure()
    ax = GeoAxes(fig, [0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    yield ax
    mpl.pyplot.close(fig)


# --- filter_points_inside_polygon ---


def test_filter_points_none_polygon():
    lons, lats = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    out_lons, out_lats = dmp.filter_points_inside_polygon(lons, lats, None)
    assert out_lons is lons
    assert out_lats is lats


def test_filter_points_inside_and_outside():
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    lons = np.array([0.5, 5.0, -1.0, 0.75, np.nan])
    lats = np.array([0.5, 5.0, 0.5, 0.25, 0.5])
    out_lons, out_lats = dmp.filter_points_inside_polygon(lons, lats, poly)
    # (0.5,0.5) and (0.75,0.25) are inside; (5,5), (-1,0.5) outside; NaN dropped
    assert list(out_lons) == [0.5, 0.75]
    assert list(out_lats) == [0.5, 0.25]


# --- categorize_ww3_files ---


def test_categorize_ww3_files():
    cats = dmp.categorize_ww3_files(
        [
            Path("WW3-ARC-15KM_202201_trck.nc"),
            Path("WW3-ANTARC-15KM_202201_trck.nc"),
            Path("WW3-IRIGLOB-7M_202201_trck.nc"),
            Path("WW3-unknown_202201_trck.nc"),
        ]
    )
    assert len(cats["arctic"]) == 1
    assert len(cats["antarctic"]) == 1
    assert len(cats["midlat"]) == 1
    assert len(cats["other"]) == 1


# --- load_ww3_points_from_files ---


def test_load_ww3_points_empty():
    lons, lats = dmp.load_ww3_points_from_files([])
    assert lons.size == 0
    assert lats.size == 0


def test_load_ww3_points_with_polygon(monkeypatch):
    def fake_open_mfdataset(_, **__):
        return _ww3_ds()

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    lons, lats = dmp.load_ww3_points_from_files(
        [Path("WW3-ARC.nc")], polygon_filter=poly
    )
    # all three points are outside the (0..10) box
    assert lons.size == 0
    assert lats.size == 0


def test_load_ww3_points_no_polygon(monkeypatch):
    def fake_open_mfdataset(_, **__):
        return _ww3_ds()

    monkeypatch.setattr(xr, "open_mfdataset", fake_open_mfdataset)
    lons, lats = dmp.load_ww3_points_from_files([Path("WW3-ARC.nc")])
    assert lons.size == 3
    assert lats.size == 3


# --- create_sar_polygon ---


def test_create_sar_polygon_valid():
    poly = dmp.create_sar_polygon(
        np.array([-5.5, -4.5, -4.5, -5.5]), np.array([47.5, 47.5, 48.5, 48.5])
    )
    assert isinstance(poly, Polygon)
    assert poly.is_valid


def test_create_sar_polygon_too_few():
    assert (
        dmp.create_sar_polygon(np.array([-5.0, -4.0]), np.array([48.0, 48.0])) is None
    )


def test_create_sar_polygon_duplicates():
    """All-same points leave <3 unique points -> None."""
    assert (
        dmp.create_sar_polygon(
            np.array([-5.0, -5.0, -5.0]), np.array([48.0, 48.0, 48.0])
        )
        is None
    )


def test_create_sar_polygon_area_too_small():
    assert (
        dmp.create_sar_polygon(
            np.array([-5.0, -4.9, -4.9, -5.0]), np.array([48.0, 48.0, 48.1, 48.1])
        )
        is None
    )


def test_create_sar_polygon_area_too_large():
    assert (
        dmp.create_sar_polygon(
            np.array([-5.0, -1.0, -1.0, -5.0]), np.array([47.0, 47.0, 51.0, 51.0])
        )
        is None
    )


# --- compute_extent ---


def test_compute_extent_with_data():
    extent = dmp.compute_extent(
        [np.array([0.0, 1.0]), np.array([2.0, 3.0])],
        [np.array([10.0, 11.0]), np.array([12.0, 13.0])],
        margin=0.5,
    )
    assert extent == [-0.5, 3.5, 9.5, 13.5]


def test_compute_extent_drops_non_finite():
    extent = dmp.compute_extent(
        [np.array([np.nan, 0.0])], [np.array([0.0, 10.0])], margin=0.0
    )
    assert extent == [0.0, 0.0, 10.0, 10.0]


def test_compute_extent_default():
    assert dmp.compute_extent([], [], margin=0.5) == [-10, 10, 40, 70]


# --- plot_sar_tiles ---


def test_plot_sar_tiles_with_corners(geo_axes):
    """Scatters centres and draws the tile polygon without error."""
    dmp.plot_sar_tiles(geo_axes, _sar_ds(), color="steelblue", label_prefix="Intra")
    assert len(geo_axes.collections) >= 1  # the scatter
    assert len(geo_axes.lines) >= 1  # the polygon outline


def test_plot_sar_tiles_without_corners(geo_axes):
    """Early-returns after the centre scatter when corners are absent."""
    dmp.plot_sar_tiles(geo_axes, _sar_ds_no_corners(), color="red")
    assert len(geo_axes.collections) >= 1
    assert len(geo_axes.lines) == 0


def test_plot_sar_tiles_no_valid_centers(geo_axes):
    """All-NaN centres skip the scatter and (no corners) the polygons."""
    ds = xr.Dataset(
        data_vars={"oswLon": ("tiles", [np.nan]), "oswLat": ("tiles", [np.nan])}
    )
    dmp.plot_sar_tiles(geo_axes, ds, color="red")
    assert len(geo_axes.collections) == 0


# --- make_base_map ---


def test_make_base_map():
    with (
        patch.object(GeoAxes, "add_image"),
        patch.object(GeoAxes, "add_feature"),
    ):
        fig, ax = dmp.make_base_map([-10, 10, 40, 70], "Test map", tile_zoom=8)
    assert isinstance(fig, Figure)
    assert isinstance(ax, GeoAxes)
    mpl.pyplot.close(fig)


# --- main() end-to-end, offline ---


def _write_trackfile(path: Path) -> None:
    path.write_text(
        "WAVEWATCH III TRACK LOCATIONS DATA\n"
        "20250407 020615 -5.00 48.00\n"
        "20250407 020615 -4.90 48.10\n",
        encoding="utf-8",
    )


def _run_main(argv: list[str], tmp_path: Path, sar_ds: xr.Dataset) -> list[Path]:
    """Drive main() with read_osw / open_mfdataset / tile fetch mocked."""

    def fake_open_mfdataset(_, **__):
        return _ww3_ds()

    with (
        patch.object(dmp, "read_osw", return_value=(sar_ds, None)),
        patch.object(GeoAxes, "add_image"),
        patch.object(GeoAxes, "add_feature"),
        patch.object(xr, "open_mfdataset", fake_open_mfdataset),
        patch.object(sys, "argv", [dmp.__name__, *argv]),
    ):
        dmp.main()

    out = sorted(tmp_path.glob("progressive_out_*.png"))
    assert len(out) == 5
    return out


def test_main_intra_only(tmp_path):
    track = tmp_path / "track.txt"
    _write_trackfile(track)
    _run_main(
        [
            "--trackfile",
            str(track),
            "--oswfiles",
            "/fake/osw1.nc",
            "--ww3files",
            "WW3-ARC-15KM_202201_trck.nc",
            "--group",
            "intraburst",
            "--output",
            str(tmp_path / "progressive_out"),
        ],
        tmp_path,
        _sar_ds(),
    )


def test_main_both_sar_driven(tmp_path):
    track = tmp_path / "track.txt"
    _write_trackfile(track)
    _run_main(
        [
            "--trackfile",
            str(track),
            "--oswfiles",
            "/fake/osw1.nc",
            "--ww3files",
            "WW3-ARC-15KM_202201_trck.nc",
            "WW3-IRIGLOB-7M_202201_trck.nc",
            "--group",
            "both",
            "--sar-driven",
            "--output",
            str(tmp_path / "progressive_out"),
        ],
        tmp_path,
        _sar_ds(),
    )
