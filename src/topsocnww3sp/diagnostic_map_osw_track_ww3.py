#!/usr/bin/env python3
"""Diagnostic map for OSW tiles, trackfile and WW3 spectra coverage.

Option --sar-driven restricts analysis to the SAR footprint.
"""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import MultiPoint, Point, Polygon

from topsocnww3sp.count_ocn_tiles_with_ww3sp import (
    core_count_coverage,
    parse_track_file,
)
from topsocnww3sp.read_s1_osw_tops_data import read_osw
from topsocnww3sp.utils import get_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def resolve_file_list(input_paths: list[str]) -> list[Path]:
    """
    Resolves a list of input paths which can be either direct file paths
    or text files containing lists of file paths.

    Args:
        input_paths (list[str]): List of input paths. Each path can be a
            direct file path or a text file containing multiple file paths
            (one per line).

    Returns:
        list[Path]: A flattened list of resolved file paths.
    """
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
    Returns None if not enough points (less than 3)."""
    points = list(zip(lons, lats, strict=True))
    if len(points) < 3:
        return None
    multi_point = MultiPoint(points)
    hull = multi_point.convex_hull
    if hull.geom_type != "Polygon":
        return None
    return hull


def filter_track_points_inside_polygon(track_points: list, polygon: Polygon) -> list:
    """Filter a list of track dictionaries to keep only those inside the polygon."""
    if polygon is None:
        return track_points
    filtered = []
    for pt in track_points:
        lon = pt["longitude"]
        lat = pt["latitude"]
        if polygon.contains(Point(lon, lat)):
            filtered.append(pt)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map OSW tiles, Trackfile, and WW3 spectra with stats."
    )
    parser.add_argument("--trackfile", required=True)
    parser.add_argument("--oswfiles", nargs="+", required=True)
    parser.add_argument("--ww3files", nargs="+", required=True)
    parser.add_argument("--group", default="intraburst")
    parser.add_argument("--zoom", type=int, default=8)
    parser.add_argument("--output", default="map_coverage.png")
    parser.add_argument("--config", default=None, help="Path to config.yml (optional)")
    parser.add_argument(
        "--sar-driven",
        action="store_true",
        help="Restrict analysis to the SAR footprint (convex hull of OSW tile centres)",
    )
    args = parser.parse_args()

    config = get_config(path_config=args.config)

    # 1. Resolve and Load Files
    osw_paths = resolve_file_list(args.oswfiles)
    ww3_paths = resolve_file_list(args.ww3files)

    logger.info("Reading OSW data...")
    _, coords_osw = read_osw(args.group, osw_paths, dev=False)

    # Compute SAR footprint if requested
    sar_polygon = None
    if args.sar_driven:
        logger.info("Computing SAR footprint (convex hull) from tile centres...")
        sar_polygon = build_convex_hull_from_points(
            coords_osw["lon_osw"], coords_osw["lat_osw"]
        )
        if sar_polygon is None:
            logger.warning(
                "Could not build convex hull (less than 3 points). "
                "Disabling SAR-driven filtering."
            )
            sar_polygon = None

    logger.info("Loading Trackfile...")
    track_points = parse_track_file(args.trackfile)

    logger.info("Loading WW3 data and calculating coverage...")
    ds_ww3 = xr.open_mfdataset(
        ww3_paths,
        combine="nested",
        concat_dim="time",
        data_vars="all",
    )
    ww3_lons_orig = ds_ww3.longitude.to_numpy().flatten()
    ww3_lats_orig = ds_ww3.latitude.to_numpy().flatten()
    ww3_times_orig = pd.to_datetime(ds_ww3.time.to_numpy())

    # Apply SAR-driven filtering if requested
    if sar_polygon is not None:
        logger.info("Filtering trackfile points inside SAR footprint...")
        orig_track_count = len(track_points)
        track_points = filter_track_points_inside_polygon(track_points, sar_polygon)
        logger.info(
            "Kept %d/%d track points inside SAR footprint.",
            len(track_points),
            orig_track_count,
        )

        logger.info("Filtering WW3 grid points inside SAR footprint...")
        orig_ww3_count = len(ww3_lons_orig)
        # Apply mask to original arrays
        mask_ww3 = np.array(
            [
                sar_polygon.contains(Point(lon, lat))
                for lon, lat in zip(ww3_lons_orig, ww3_lats_orig, strict=True)
            ]
        )
        ww3_lons = ww3_lons_orig[mask_ww3]
        ww3_lats = ww3_lats_orig[mask_ww3]
        ww3_times = ww3_times_orig[mask_ww3]
        logger.info(
            "Kept %d/%d WW3 points inside SAR footprint.",
            len(ww3_lons),
            orig_ww3_count,
        )
    else:
        # No filtering
        ww3_lons = ww3_lons_orig
        ww3_lats = ww3_lats_orig
        ww3_times = ww3_times_orig

    # 2. Run logic to get summary and hit counts per point
    summary_lines, results = core_count_coverage(
        track_points, ww3_lons, ww3_lats, ww3_times, pathconfig=args.config
    )
    summary_text = "\n".join(summary_lines)

    # Prepare data for plotting
    t_lons = np.array([p["longitude"] for p in track_points])
    t_lats = np.array([p["latitude"] for p in track_points])
    t_hits = np.array([results.get(p["line_idx"], 0) for p in track_points])

    # ---------------------------------------------------------
    # FIGURE 1: OVERVIEW MAP (Coverage layers)
    # ---------------------------------------------------------
    request = cimgt.GoogleTiles(style="satellite")
    fig1 = plt.figure(figsize=(18, 12), dpi=110)
    ax1 = fig1.add_axes([0.05, 0.1, 0.65, 0.8], projection=request.crs)

    all_lons = np.concatenate([coords_osw["lon_osw"], t_lons, ww3_lons])
    all_lats = np.concatenate([coords_osw["lat_osw"], t_lats, ww3_lats])
    extent = [
        np.min(all_lons) - 0.3,
        np.max(all_lons) + 0.3,
        np.min(all_lats) - 0.3,
        np.max(all_lats) + 0.3,
    ]
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.add_image(request, args.zoom)

    ax1.scatter(
        coords_osw["lon_osw"],
        coords_osw["lat_osw"],
        transform=ccrs.PlateCarree(),
        s=16,
        c="magenta",
        marker="s",
        edgecolors="white",
        label=f"S1 OSW ({len(coords_osw['lon_osw'])})",
    )
    ax1.scatter(
        t_lons,
        t_lats,
        transform=ccrs.PlateCarree(),
        s=14,
        c="yellow",
        marker="D",
        edgecolors="black",
        label=f"Trackfile ({len(t_lons)})",
        alpha=0.6,
    )
    ax1.scatter(
        ww3_lons,
        ww3_lats,
        transform=ccrs.PlateCarree(),
        s=8,
        c="cyan",
        alpha=0.3,
        label=f"WW3 ({len(ww3_lons)})",
    )

    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    title = f"S1 OCN/WW3 Overview - {args.group}"
    if args.sar_driven and sar_polygon is not None:
        title += " (SAR-driven filtering)"
    ax1.set_title(title, fontsize=14)

    # Summary box on the right
    fig1.text(
        0.72,
        0.5,
        summary_text,
        fontsize=8,
        family="monospace",
        verticalalignment="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    out1 = args.output
    plt.savefig(out1, bbox_inches="tight")
    logger.info("Saved Figure 1 to %s", out1)

    # ---------------------------------------------------------
    # FIGURE 2: HIT COUNT GRADIENT MAP
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 10), dpi=110)
    ax2 = plt.axes(projection=request.crs)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.add_image(request, args.zoom)

    mask_zero = t_hits == 0
    mask_pos = t_hits > 0

    ax2.scatter(
        t_lons[mask_zero],
        t_lats[mask_zero],
        transform=ccrs.PlateCarree(),
        s=30,
        c="red",
        marker="o",
        edgecolors="black",
        label="0 WW3 Spectra",
    )

    if np.any(mask_pos):
        sc = ax2.scatter(
            t_lons[mask_pos],
            t_lats[mask_pos],
            transform=ccrs.PlateCarree(),
            s=30,
            c=t_hits[mask_pos],
            cmap="viridis",
            edgecolors="white",
            linewidth=0.5,
            label=">0 WW3 Spectra",
        )
        cbar = plt.colorbar(sc, ax=ax2, orientation="vertical", pad=0.02, aspect=30)
        cbar.set_label("Number of associated WW3 spectra", fontsize=12)

    ax2.legend(loc="lower left")
    title2 = (
        f"Track Points: Hit Count Distribution\n(Red = No spectra within "
        f"{config['DISTANCE_THRESHOLD_KM']}km/{config['TIME_THRESHOLD_MINUTES']}min)"
    )
    if args.sar_driven and sar_polygon is not None:
        title2 += " [SAR-driven]"
    ax2.set_title(title2, fontsize=14)

    out2 = args.output.replace(".png", "_hit_distribution.png")
    plt.savefig(out2, bbox_inches="tight")
    logger.info("Saved Figure 2 to %s", out2)

    plt.show()


if __name__ == "__main__":
    main()
