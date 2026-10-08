#!/usr/bin/env python3
"""Shared helpers for the diagnostic mapping scripts (file lists, SAR footprint)."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

logger = logging.getLogger(__name__)


def resolve_file_list(input_paths: list[str]) -> list[Path]:
    """Resolve input paths that are either direct file paths or .txt listings.

    Each entry of ``input_paths`` is taken as a literal path unless it ends in
    ``.txt``, in which case its non-empty, non-comment lines are treated as one
    path per line and appended to the result."""
    resolved_files = []
    for path_str in input_paths:
        if path_str.endswith(".txt"):
            path_obj = Path(path_str)
            if not path_obj.exists():
                logger.error("Listing file not found: %s", path_str)
                continue
            with path_obj.open(encoding="utf-8") as f:
                files_from_txt = [
                    Path(line.strip())
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
                resolved_files.extend(files_from_txt)
        else:
            resolved_files.append(Path(path_str))
    return resolved_files


def build_convex_hull_from_points(lons: np.ndarray, lats: np.ndarray) -> Polygon | None:
    """Build a convex hull polygon from longitude/latitude points.

    Non-finite coordinates are dropped first; returns ``None`` when fewer than
    3 finite points remain or the hull degenerates to a non-polygon."""
    lons = np.asarray(lons, dtype=np.float64).ravel()
    lats = np.asarray(lats, dtype=np.float64).ravel()
    mask = np.isfinite(lons) & np.isfinite(lats)
    points = list(zip(lons[mask], lats[mask], strict=True))
    if len(points) < 3:
        return None
    hull = MultiPoint(points).convex_hull
    if hull.geom_type != "Polygon":
        return None
    return hull


def filter_track_points_inside_polygon(
    track_points: list[dict[str, Any]], polygon: Polygon | None
) -> list[dict[str, Any]]:
    """Filter track dictionaries to keep only points inside the given polygon.

    Points whose longitude/latitude are non-finite (or absent) cannot be tested
    and are dropped; a ``None`` polygon passes every finite point through."""
    if not isinstance(polygon, Polygon):
        if polygon is None:
            return list(track_points)
        return []  # pragma: no cover - degenerate geometry guard only
    filtered = []
    for pt in track_points:
        lon = pt["longitude"]
        lat = pt["latitude"]
        try:
            finite = bool(np.isfinite(float(lon)) and np.isfinite(float(lat)))
        except (TypeError, ValueError):  # pragma: no cover - malformed guard only
            continue
        if finite and polygon.contains(Point(float(lon), float(lat))):
            filtered.append(pt)
    return filtered
