#!/usr/bin/env python3
"""Unit tests for the shared diagnostic-mapping helpers in map_helpers."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from topsocnww3sp.map_helpers import (
    build_convex_hull_from_points,
    filter_track_points_inside_polygon,
    resolve_file_list,
)

# --- resolve_file_list ---


def test_resolve_file_list_direct_paths():
    """Non-.txt entries are returned as-is."""
    result = resolve_file_list(["/data/a.nc", "/data/b.nc"])
    assert result == [Path("/data/a.nc"), Path("/data/b.nc")]


def test_resolve_file_list_from_txt_listing(tmp_path):
    """.txt entries are expanded, skipping comments and blank lines."""
    listing = tmp_path / "listing.txt"
    listing.write_text(
        "# a comment\n\n/data/a.nc\n   /data/b.nc   \n/data/c.nc\n", encoding="utf-8"
    )
    result = resolve_file_list(["/data/a.nc", str(listing)])
    assert result == [
        Path("/data/a.nc"),
        Path("/data/a.nc"),
        Path("/data/b.nc"),
        Path("/data/c.nc"),
    ]


def test_resolve_file_list_missing_listing(tmp_path):
    """A missing .txt listing is skipped without raising."""
    missing = tmp_path / "nope.txt"
    result = resolve_file_list(["/data/a.nc", str(missing)])
    assert result == [Path("/data/a.nc")]


# --- build_convex_hull_from_points ---


def test_convex_hull_returns_polygon():
    """Three or more non-collinear points yield a Polygon hull."""
    hull = build_convex_hull_from_points(
        np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert isinstance(hull, Polygon)
    assert hull.geom_type == "Polygon"


def test_convex_hull_too_few_points():
    """Fewer than 3 finite points returns None."""
    assert (
        build_convex_hull_from_points(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        is None
    )


def test_convex_hull_drops_nan():
    """Non-finite coordinates are dropped before the hull is built."""
    lons = np.array([0.0, 1.0, 1.0, 0.0, np.nan])
    lats = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    hull = build_convex_hull_from_points(lons, lats)
    assert isinstance(hull, Polygon)


def test_convex_hull_collinear_returns_none():
    """Collinear points degenerate to a line, not a polygon."""
    assert (
        build_convex_hull_from_points(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
        )
        is None
    )


# --- filter_track_points_inside_polygon ---


@pytest.fixture
def square():
    """A unit square polygon around (0, 0)."""
    return Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def test_filter_track_none_polygon_passthrough():
    """A None polygon passes every point through (as a new list)."""
    points = [{"longitude": 5.0, "latitude": 5.0}]
    result = filter_track_points_inside_polygon(points, None)
    assert result == points
    assert result is not points


def test_filter_track_inside_and_outside(square):
    """Only points strictly inside the polygon are kept."""
    points = [
        {"longitude": 0.5, "latitude": 0.5},  # inside
        {"longitude": 5.0, "latitude": 5.0},  # outside
        {"longitude": -1.0, "latitude": 0.5},  # outside
    ]
    result = filter_track_points_inside_polygon(points, square)
    assert result == [{"longitude": 0.5, "latitude": 0.5}]


def test_filter_track_drops_non_finite(square):
    """Points with non-finite coordinates are dropped."""
    points = [
        {"longitude": 0.5, "latitude": 0.5},  # kept
        {"longitude": np.nan, "latitude": 0.5},  # dropped
        {"longitude": 0.5, "latitude": float("inf")},  # dropped
    ]
    result = filter_track_points_inside_polygon(points, square)
    assert result == [{"longitude": 0.5, "latitude": 0.5}]
