#!/usr/bin/env python3
"""Progressive diagnostic maps: SAR intra, intra+inter, +track, +WW3 ARCTIC, +WW3 MIDLAT.

Usage:
  diagnostic_map_progressive.py --trackfile trackfile.txt --oswfiles listing_osw.txt --ww3files listing_ww3.txt --group intraburst --output prefix --sar-driven
"""

import argparse
import logging
from pathlib import Path
from typing import TypedDict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
from shapely.geometry import MultiPoint, Point
from shapely.geometry import Polygon as sPolygon

from topsocnww3sp.count_ocn_tiles_with_ww3sp import parse_track_file
from topsocnww3sp.map_helpers import (
    build_convex_hull_from_points,
    filter_track_points_inside_polygon,
    resolve_file_list,
)
from topsocnww3sp.read_s1_osw_tops_data import read_osw

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MapDef(TypedDict):
    """Specification of one progressive map: name, title and SAR/track/WW3 layers."""

    name: str
    title: str
    layers: list[tuple[str, str, str]]


def filter_points_inside_polygon(
    lons: np.ndarray, lats: np.ndarray, polygon: sPolygon | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of lons/lats that lie inside the polygon."""
    if polygon is None:
        return lons, lats
    mask_valid = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[mask_valid]
    lats = lats[mask_valid]
    mask = np.array(
        [polygon.contains(Point(lo, la)) for lo, la in zip(lons, lats, strict=True)]
    )
    return lons[mask], lats[mask]


def categorize_ww3_files(file_paths: list[Path]) -> dict[str, list[Path]]:
    """Categorize WW3 NetCDF files by grid name based on path content."""
    categories: dict[str, list[Path]] = {
        "arctic": [],
        "antarctic": [],
        "midlat": [],
        "other": [],
    }
    for p in file_paths:
        path_str = str(p)
        if "ARC" in path_str and "ANTARC" not in path_str:
            categories["arctic"].append(p)
        elif "ANTARC" in path_str:
            categories["antarctic"].append(p)
        elif "IRIGLOB" in path_str:
            categories["midlat"].append(p)
        else:
            categories["other"].append(p)
    return categories


def load_ww3_points_from_files(
    file_list: list[Path], polygon_filter: sPolygon | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load longitude/latitude points from a list of WW3 files."""
    if not file_list:
        return np.array([]), np.array([])
    ds = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim="time",
        data_vars="all",
    )
    lons = ds.longitude.to_numpy().flatten()
    lats = ds.latitude.to_numpy().flatten()
    mask = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[mask]
    lats = lats[mask]
    if polygon_filter is not None:
        lons, lats = filter_points_inside_polygon(lons, lats, polygon_filter)
    return lons, lats


def make_base_map(
    extent: list[float], title: str, tile_zoom: int = 8
) -> tuple[Figure, GeoAxes]:
    """Create base map with satellite imagery and features."""
    tiler = cimgt.GoogleTiles(style="satellite")
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={"projection": tiler.crs})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, tile_zoom)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white", zorder=2)
    ax.add_feature(
        cfeature.BORDERS, linewidth=0.4, edgecolor="white", linestyle=":", zorder=2
    )
    ax.add_feature(
        cfeature.LAKES, edgecolor="white", facecolor="none", linewidth=0.5, zorder=2
    )
    ax.add_feature(cfeature.RIVERS, edgecolor="white", linewidth=0.4, zorder=2)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="white", alpha=0.6, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=14, pad=20)
    return fig, ax


def create_sar_polygon(lons: np.ndarray, lats: np.ndarray) -> sPolygon | None:
    """Create a valid Shapely polygon from corner coordinates."""
    # Filter invalid values
    mask = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[mask]
    lats = lats[mask]
    if len(lons) < 3:
        return None
    points = list(zip(lons, lats, strict=True))
    # Remove consecutive duplicates
    unique_points: list[tuple[float, float]] = []
    for p in points:
        if not unique_points or p != unique_points[-1]:
            unique_points.append(p)
    if len(unique_points) < 3:
        return None

    # Try multiple orders
    orders = [
        [0, 1, 2, 3, 0],
        [0, 1, 3, 2, 0],
        [0, 2, 1, 3, 0],
    ]
    for order in orders:
        pts = [unique_points[i] for i in order if i < len(unique_points)]
        if len(pts) >= 4:
            poly = sPolygon(pts)
            if poly.is_valid and not poly.is_empty and poly.is_simple:
                # Check area is reasonable (not too large)
                area = poly.area
                if area < 0.1 or area > 10.0:
                    # Too small or too large (could be a rectangle spanning globe)
                    pass
                else:
                    return poly
    # If all orders fail, try convex hull
    multi = MultiPoint(unique_points)
    hull = multi.convex_hull
    if hull.is_valid and hull.geom_type == "Polygon":
        area = hull.area
        if 0.1 < area < 10.0:
            return hull
    return None


def plot_sar_tiles(
    ax: GeoAxes,
    ds: xr.Dataset,
    color: str,
    alpha_fill: float = 0.25,
    alpha_edge: float = 0.6,
    label_prefix: str = "",
    zorder: int = 3,
) -> None:
    """Plot SAR tile footprints and center points on the given cartopy axis."""
    # -- Centres --
    lons = ds["oswLon"].values.flatten()
    lats = ds["oswLat"].values.flatten()
    mask = np.isfinite(lons) & np.isfinite(lats)
    n_centers = np.sum(mask)
    if n_centers > 0:
        label = f"{label_prefix} ({n_centers})" if label_prefix else None
        ax.scatter(
            lons[mask],
            lats[mask],
            s=15,
            color=color,
            marker="o",
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            zorder=zorder + 2,
            label=label,
        )

    # -- Polygones --
    if "oswLongitudeCorner" not in ds:
        return

    lon_corners = ds["oswLongitudeCorner"].values
    lat_corners = ds["oswLatitudeCorner"].values

    # Reshape to (n_polygons, 4); arrays already shaped that way are kept as-is.
    if lon_corners.ndim != 2:
        lon_corners = lon_corners.reshape(-1, 4)
        lat_corners = lat_corners.reshape(-1, 4)

    n_polys = lon_corners.shape[0]
    for i in range(n_polys):
        lons_i = lon_corners[i, :]
        lats_i = lat_corners[i, :]
        poly = create_sar_polygon(lons_i, lats_i)
        if poly is None:
            continue
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            color=color,
            alpha=alpha_fill,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
        )
        ax.plot(
            x,
            y,
            color=color,
            linewidth=0.8,
            alpha=alpha_edge,
            transform=ccrs.PlateCarree(),
            zorder=zorder + 1,
        )


def compute_extent(
    lons_list: list[np.ndarray], lats_list: list[np.ndarray], margin: float = 0.5
) -> list[float]:
    """Compute extent from a list of (lons, lats) arrays."""
    all_lons: list[np.float64] = []
    all_lats: list[np.float64] = []
    for lons, lats in zip(lons_list, lats_list, strict=True):
        mask = np.isfinite(lons) & np.isfinite(lats)
        if np.any(mask):
            all_lons.extend(lons[mask])
            all_lats.extend(lats[mask])
    if all_lons:
        return [
            np.min(all_lons) - margin,
            np.max(all_lons) + margin,
            np.min(all_lats) - margin,
            np.max(all_lats) + margin,
        ]
    return [-10, 10, 40, 70]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate progressive diagnostic maps (SAR intra, inter, track, WW3 layers)."
    )
    parser.add_argument("--trackfile", required=True)
    parser.add_argument("--oswfiles", nargs="+", required=True)
    parser.add_argument("--ww3files", nargs="+", required=True)
    parser.add_argument(
        "--group",
        default="intraburst",
        choices=["intraburst", "interburst", "both"],
        help="Which SAR group(s) to load. 'both' will load intra and inter.",
    )
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument(
        "--output",
        default="progressive_map",
        help="Base output name (without extension)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yml (optional)")
    parser.add_argument(
        "--sar-driven",
        action="store_true",
        help="Restrict all layers to SAR footprint (convex hull of tile centres).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    osw_paths = resolve_file_list(args.oswfiles)
    ww3_paths = resolve_file_list(args.ww3files)

    logger.info("Loading OSW data...")
    sar_intra_ds, _ = read_osw("intraburst", osw_paths, dev=False)
    sar_inter_ds = None
    if args.group in ("interburst", "both"):
        sar_inter_ds, _ = read_osw("interburst", osw_paths, dev=False)

    sar_polygon = None
    if args.sar_driven:
        logger.info("Computing SAR footprint (convex hull) from tile centres...")
        all_lons = sar_intra_ds["oswLon"].values.flatten()
        all_lats = sar_intra_ds["oswLat"].values.flatten()
        if sar_inter_ds is not None:
            all_lons = np.concatenate(
                [all_lons, sar_inter_ds["oswLon"].values.flatten()]
            )
            all_lats = np.concatenate(
                [all_lats, sar_inter_ds["oswLat"].values.flatten()]
            )
        sar_polygon = build_convex_hull_from_points(all_lons, all_lats)
        if sar_polygon is None:
            logger.warning(
                "Could not build convex hull. Disabling SAR-driven filtering."
            )
            sar_polygon = None

    logger.info("Loading Trackfile...")
    track_points = parse_track_file(args.trackfile)
    if sar_polygon is not None:
        track_points = filter_track_points_inside_polygon(track_points, sar_polygon)

    ww3_cats = categorize_ww3_files(ww3_paths)
    ww3_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cat, files in ww3_cats.items():
        if files:
            logger.info(
                "Loading WW3 points for category '%s' (%d files)", cat, len(files)
            )
            lons, lats = load_ww3_points_from_files(files, polygon_filter=sar_polygon)
            ww3_points[cat] = (lons, lats)
        else:
            ww3_points[cat] = (np.array([]), np.array([]))

    t_lons = np.array([p["longitude"] for p in track_points])
    t_lats = np.array([p["latitude"] for p in track_points])
    mask_t = np.isfinite(t_lons) & np.isfinite(t_lats)
    t_lons = t_lons[mask_t]
    t_lats = t_lats[mask_t]

    lons_list = [
        sar_intra_ds["oswLon"].values.flatten(),
        sar_inter_ds["oswLon"].values.flatten()
        if sar_inter_ds is not None
        else np.array([]),
        t_lons,
    ]
    lats_list = [
        sar_intra_ds["oswLat"].values.flatten(),
        sar_inter_ds["oswLat"].values.flatten()
        if sar_inter_ds is not None
        else np.array([]),
        t_lats,
    ]
    for lons, lats in ww3_points.values():
        lons_list.append(lons)
        lats_list.append(lats)
    extent = compute_extent(lons_list, lats_list, margin=0.5)

    # --- Generate maps ---
    sar_intra_color = "steelblue"
    sar_inter_color = "darkorange"
    track_color = "yellow"
    ww3_arctic_color = "limegreen"
    ww3_midlat_color = "cyan"

    map_defs: list[MapDef] = [
        {
            "name": "map1_sar_intra",
            "title": "Layer 1: SAR Intraburst",
            "layers": [("intra", sar_intra_color, "Intraburst")],
        },
        {
            "name": "map2_sar_intra_inter",
            "title": "Layer 2: SAR Intraburst + Interburst",
            "layers": [
                ("intra", sar_intra_color, "Intraburst"),
                ("inter", sar_inter_color, "Interburst"),
            ],
        },
        {
            "name": "map3_sar_intra_inter_track",
            "title": "Layer 3: SAR + Trackfile",
            "layers": [
                ("intra", sar_intra_color, "Intraburst"),
                ("inter", sar_inter_color, "Interburst"),
                ("track", track_color, "Trackfile"),
            ],
        },
        {
            "name": "map4_sar_intra_inter_track_ww3_arctic",
            "title": "Layer 4: SAR + Trackfile + WW3 Arctic",
            "layers": [
                ("intra", sar_intra_color, "Intraburst"),
                ("inter", sar_inter_color, "Interburst"),
                ("track", track_color, "Trackfile"),
                ("ww3_arctic", ww3_arctic_color, "WW3 Arctic"),
            ],
        },
        {
            "name": "map5_sar_intra_inter_track_ww3_arctic_midlat",
            "title": "Layer 5: SAR + Trackfile + WW3 Arctic + WW3 Mid-Lat",
            "layers": [
                ("intra", sar_intra_color, "Intraburst"),
                ("inter", sar_inter_color, "Interburst"),
                ("track", track_color, "Trackfile"),
                ("ww3_arctic", ww3_arctic_color, "WW3 Arctic"),
                ("ww3_midlat", ww3_midlat_color, "WW3 Mid-Lat"),
            ],
        },
    ]

    for map_def in map_defs:
        logger.info("Generating %s ...", map_def["name"])
        fig, ax = make_base_map(extent, map_def["title"], tile_zoom=args.zoom)

        for layer_type, color, base_label in map_def["layers"]:
            if layer_type == "intra" and sar_intra_ds is not None:
                plot_sar_tiles(
                    ax, sar_intra_ds, color=color, label_prefix=base_label, zorder=3
                )
            elif layer_type == "inter" and sar_inter_ds is not None:
                plot_sar_tiles(
                    ax, sar_inter_ds, color=color, label_prefix=base_label, zorder=2
                )
            elif layer_type == "track":
                label = f"{base_label} ({len(t_lons)})"
                if len(t_lons) > 0:
                    ax.scatter(
                        t_lons,
                        t_lats,
                        s=30,
                        c=color,
                        marker="D",
                        alpha=0.6,
                        transform=ccrs.PlateCarree(),
                        label=label,
                        zorder=4,
                        rasterized=True,
                        edgecolors="black",
                    )
            elif layer_type == "ww3_arctic":
                lons, lats = ww3_points.get("arctic", (np.array([]), np.array([])))
                label = f"{base_label} ({len(lons)})"
                if len(lons) > 0:
                    ax.scatter(
                        lons,
                        lats,
                        s=40,
                        c=color,
                        marker=".",
                        alpha=0.4,
                        transform=ccrs.PlateCarree(),
                        label=label,
                        zorder=5,
                        rasterized=True,
                    )
            elif layer_type == "ww3_midlat":
                lons, lats = ww3_points.get("midlat", (np.array([]), np.array([])))
                label = f"{base_label} ({len(lons)})"
                if len(lons) > 0:
                    ax.scatter(
                        lons,
                        lats,
                        s=40,
                        c=color,
                        marker=".",
                        alpha=0.4,
                        transform=ccrs.PlateCarree(),
                        label=label,
                        zorder=6,
                        rasterized=True,
                    )

        ax.legend(loc="lower left", fontsize=9)
        if args.sar_driven and sar_polygon is not None:
            ax.set_title(map_def["title"] + " (SAR-driven)", fontsize=14, pad=20)
        out_path = Path(args.output + "_" + map_def["name"] + ".png")
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", out_path)

    logger.info("All maps generated successfully.")


if __name__ == "__main__":
    main()
