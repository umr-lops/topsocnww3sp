#!/usr/bin/env python3
"""Diagnostic map for OSW tiles, trackfile and WW3 spectra coverage."""

import argparse
import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
            with path_obj.open() as f:
                files_from_txt = [
                    Path(line.strip())
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
                resolved_files.extend(files_from_txt)
        else:
            resolved_files.append(Path(path_str))
    return resolved_files


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
    args = parser.parse_args()
    config = get_config(path_config=args.config)
    # 1. Resolve and Load Files
    osw_paths = resolve_file_list(args.oswfiles)
    ww3_paths = resolve_file_list(args.ww3files)

    logger.info("Reading OSW data...")
    _, coords_osw = read_osw(args.group, osw_paths, dev=False)

    logger.info("Loading Trackfile...")
    track_points = parse_track_file(args.trackfile)

    logger.info("Loading WW3 data and calculating coverage...")
    ds_ww3 = xr.open_mfdataset(ww3_paths, combine="nested", concat_dim="time")
    ww3_lons = ds_ww3.longitude.to_numpy().flatten()
    ww3_lats = ds_ww3.latitude.to_numpy().flatten()
    ww3_times = pd.to_datetime(ds_ww3.time.to_numpy())

    # 2. Run logic to get summary and hit counts per point
    summary_lines, results = core_count_coverage(
        track_points, ww3_lons, ww3_lats, ww3_times, pathconfig=args.config
    )
    summary_text = "\n".join(summary_lines)

    # Prepare data for plotting
    t_lons = np.array([p["longitude"] for p in track_points])
    t_lats = np.array([p["latitude"] for p in track_points])
    # Extract hit counts in same order as lons/lats
    t_hits = np.array([results[p["line_idx"]] for p in track_points])

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
    ax1.set_title(f"S1 OCN/WW3 Overview - {args.group}", fontsize=14)

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

    # Mask for zero vs positive
    mask_zero = t_hits == 0
    mask_pos = t_hits > 0

    # 1. Plot 0 hits as Red
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

    # 2. Plot >0 hits with graduate colors (using 'viridis' or 'plasma')
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

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax2, orientation="vertical", pad=0.02, aspect=30)
        cbar.set_label("Number of associated WW3 spectra", fontsize=12)

    ax2.legend(loc="lower left")
    ax2.set_title(
        f"Track Points: Hit Count Distribution\n(Red = No spectra within {config['DISTANCE_THRESHOLD_KM']}km/{config['TIME_THRESHOLD_MINUTES']}min)",
        fontsize=14,
    )

    out2 = args.output.replace(".png", "_hit_distribution.png")
    plt.savefig(out2, bbox_inches="tight")
    logger.info("Saved Figure 2 to %s", out2)

    plt.show()


if __name__ == "__main__":
    main()
