#!/usr/bin/env python3
"""Render a WW3 integrated-parameter map from a topsocnww3sp L2C SAFE.

Plots, with cartopy and matplotlib, a single geographic map showing the WW3
integrated (FIELD) parameter sampled at the WW3 track points of **every**
subswath of a L2C SAFE (e.g. ``hs``, the significant wave height), together
with each SAR subswath footprint. The WW3 points of all subswaths share one
colour scale (the plotted variable) while each subswath footprint gets its own
outline colour. The resulting PNG is intended as documentation evidence for
the co-localization.

Example:
    python end2end/plot_ww3_hs_map_l2c.py \\
        --input-l2c-safe /path/to/SAFE_output_dir \\
        --output-png ww3_hs_map.png
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray as xr
from cartopy import feature as cfeature
from shapely.geometry import MultiPoint, Point

logger = logging.getLogger(__name__)

WW3_GROUP_CANDIDATES = ("WW3", "WW3_intraburst", "WW3_interburst")
SAR_GROUP = "SAR_intraburst"
CORNER_VARS = ("oswLongitudeCorner", "oswLatitudeCorner")
# One outline colour per subswath footprint (dashed)
SUBSWATH_COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


@dataclass
class Subswath:
    """WW3 sample points and SAR footprint for one L2C file (one subswath)."""

    name: str
    lons: np.ndarray
    lats: np.ndarray
    values: np.ndarray
    units: str
    long_name: str
    footprint: MultiPoint

    @property
    def n_points(self) -> int:
        """Number of finite WW3 sample points."""
        return int(self.lons.size)


def collect_l2c_files(path: Path) -> list[Path]:
    """Resolve the list of L2C files to plot from a file or a directory.

    Args:
        path: A single L2C ``.nc`` file, or a directory (e.g. the output SAFE
            directory) containing the per-subswath L2C files.

    Returns:
        Sorted list of L2C file paths.

    Raises:
        FileNotFoundError: If no L2C file can be located.
    """
    if path.is_file():
        return [path]
    files = sorted(path.glob("s1a-*_v*.nc")) or sorted(path.glob("*.nc"))
    if not files:
        msg = f"No L2C .nc file found in {path}"
        raise FileNotFoundError(msg)
    logger.info("Found %d L2C file(s) in %s", len(files), path)
    return files


def find_ww3_group(l2c_file: Path) -> str:
    """Return the name of the WW3 group present in the L2C file.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        The WW3 group name (``WW3`` or ``WW3_{intraburst,interburst}``)

    Raises:
        KeyError: If no WW3 group is found.
    """
    with netCDF4.Dataset(l2c_file) as nc:  # pylint: disable=no-member
        groups = set(nc.groups)
    for group in WW3_GROUP_CANDIDATES:
        if group in groups:
            return group
    msg = f"No WW3 group found in {l2c_file} (tried {WW3_GROUP_CANDIDATES})"
    raise KeyError(msg)


def subswath_name(l2c_file: Path) -> str:
    """Extract the subswath (e.g. ``IW1``) from an L2C filename.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        The upper-cased subswath token, or the file stem as a fallback.
    """
    parts = l2c_file.stem.split("-")
    if len(parts) > 1:
        return parts[1].upper()
    return l2c_file.stem


def sar_footprint(l2c_file: Path) -> MultiPoint:
    """Build the SAR subswath footprint from the tile corner coordinates.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        A MultiPoint of all tile corner coordinates (may be empty).
    """
    try:
        sar = xr.open_dataset(l2c_file, group=SAR_GROUP)
    except (KeyError, OSError):
        return MultiPoint()
    try:
        lon = sar[CORNER_VARS[0]].to_numpy().ravel()
        lat = sar[CORNER_VARS[1]].to_numpy().ravel()
        lon = lon[np.isfinite(lon)]
        lat = lat[np.isfinite(lat)]
    except KeyError:
        return MultiPoint()
    finally:
        sar.close()
    if lon.size == 0:
        return MultiPoint()
    return MultiPoint([Point(lo, la) for lo, la in zip(lon, lat, strict=True)])


def load_subswath(l2c_file: Path, variable: str) -> Subswath:
    """Load the WW3 sample points and SAR footprint for one subswath.

    Args:
        l2c_file: Path to the L2C file
        variable: WW3 integrated-parameter variable name to plot (e.g. ``hs``)

    Returns:
        A Subswath with finite (lon, lat, value) arrays and the footprint.

    Raises:
        KeyError: If the variable is not in the WW3 group.
    """
    ww3 = xr.open_dataset(l2c_file, group=find_ww3_group(l2c_file))
    try:
        if variable not in ww3.data_vars:
            available = [v for v in ww3.data_vars if v not in ("longitude", "latitude")]
            msg = (
                f"Variable {variable!r} not in the WW3 group of {l2c_file.name} "
                f"(available: {available})"
            )
            raise KeyError(msg)
        lons = ww3["longitude"].to_numpy().astype("f8")
        lats = ww3["latitude"].to_numpy().astype("f8")
        values = ww3[variable].to_numpy().astype("f4")
        units = ww3[variable].attrs.get("units", "")
        long_name = ww3[variable].attrs.get("long_name", variable)
    finally:
        ww3.close()
    valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(values)
    return Subswath(
        name=subswath_name(l2c_file),
        lons=lons[valid],
        lats=lats[valid],
        values=values[valid],
        units=units,
        long_name=long_name,
        footprint=sar_footprint(l2c_file),
    )


def make_plot(l2c_files: list[Path], variable: str, output_png: Path) -> None:
    """Create a geographic map of a WW3 integrated parameter for a SAFE.

    All subswaths of the SAFE are drawn on a single map, each with its own
    footprint outline; the WW3 track points share one colour scale.

    Args:
        l2c_files: Paths to the per-subswath L2C files.
        variable: WW3 integrated-parameter variable name to plot (e.g. ``hs``).
        output_png: Destination PNG path.

    Raises:
        ValueError: If no finite value is available to plot.
    """
    subs: list[Subswath] = []
    for f in l2c_files:
        sub = load_subswath(f, variable)
        if sub.n_points:
            subs.append(sub)
    if not subs:
        msg = f"No finite {variable} values to plot in {l2c_files}"
        raise ValueError(msg)

    all_lons = np.concatenate([s.lons for s in subs])
    all_lats = np.concatenate([s.lats for s in subs])
    all_vals = np.concatenate([s.values for s in subs])

    # Map extent spans all WW3 points and all subswath footprints.
    min_lon, max_lon = float(all_lons.min()), float(all_lons.max())
    min_lat, max_lat = float(all_lats.min()), float(all_lats.max())
    for s in subs:
        if s.footprint.is_empty:
            continue
        fminx, fminy, fmaxx, fmaxy = s.footprint.bounds
        min_lon = min(min_lon, fminx)
        max_lon = max(max_lon, fmaxx)
        min_lat = min(min_lat, fminy)
        max_lat = max(max_lat, fmaxy)
    pad_lon = max(0.5, 0.05 * (max_lon - min_lon))
    pad_lat = max(0.5, 0.05 * (max_lat - min_lat))
    extent = [
        min_lon - pad_lon,
        max_lon + pad_lon,
        min_lat - pad_lat,
        max_lat + pad_lat,
    ]

    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Land / coastlines
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d5", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#8a8a8a")
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="#cccccc",
        alpha=0.7,
    )

    # SAR subswath footprints, one distinct colour each
    for i, s in enumerate(subs):
        if s.footprint.is_empty:
            continue
        hull = s.footprint.convex_hull
        xs, ys = hull.exterior.xy
        color = SUBSWATH_COLORS[i % len(SUBSWATH_COLORS)]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{s.name} footprint ({s.n_points} pts)",
        )

    # WW3 track points (all subswaths) coloured by the plotted variable
    scatter = ax.scatter(
        all_lons,
        all_lats,
        c=all_vals,
        cmap="viridis",
        s=24,
        edgecolor="k",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
        label=f"WW3 track points ({all_lons.size})",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(f"{variable} [{subs[0].units}]")

    ax.set_title(
        f"WW3 {subs[0].long_name}\n{len(subs)} subswaths, {all_lons.size} track points",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_png)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Plot a WW3 integrated-parameter map from an L2C SAFE"
    )
    parser.add_argument(
        "--input-l2c-safe",
        required=True,
        help="Path to a single L2C .nc file or a directory containing the "
        "per-subswath L2C files of a SAFE (all are drawn on one map)",
    )
    parser.add_argument(
        "--output-png",
        required=True,
        help="Destination PNG file",
    )
    parser.add_argument(
        "--variable",
        default="hs",
        help="WW3 integrated-parameter variable to plot (default: hs)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    files = collect_l2c_files(Path(args.input_l2c_safe))
    make_plot(files, args.variable, Path(args.output_png))


if __name__ == "__main__":
    main()
