#!/usr/bin/env python3
"""Ensure the OSW/track/WW3 map script reuses the shared map_helpers (no duplication)."""

import topsocnww3sp.diagnostic_map_osw_track_ww3 as m
from topsocnww3sp import map_helpers


def test_shared_helpers_are_reexported():
    """The map script imports its helpers from map_helpers (same objects)."""
    module_vars = vars(m)
    assert module_vars["resolve_file_list"] is map_helpers.resolve_file_list
    assert (
        module_vars["build_convex_hull_from_points"]
        is map_helpers.build_convex_hull_from_points
    )
    assert (
        module_vars["filter_track_points_inside_polygon"]
        is map_helpers.filter_track_points_inside_polygon
    )
