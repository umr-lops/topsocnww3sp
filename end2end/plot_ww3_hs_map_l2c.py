#!/usr/bin/env python3
"""Render a WW3 integrated-parameter map from a topsocnww3sp L2C file.

Plots, with cartopy and matplotlib, a geographic map showing the WW3
integrated (FIELD) parameter sampled at the WW3 track points (e.g. ``hs``, the
significant wave height) together with the SAR subswath footprint extracted in
the L2C product. The resulting PNG is intended to be used as documentation
evidence for the co-localization.

Example:
    python end2end/plot_ww3_hs_map_l2c.py \\
        --input-l2c-safe /path/to/s1a-iw1-osw-..._v0.1.nc \\
        --output-png ww3_hs_map.png
"""

from __future__ import annotations

import argparse
import logging
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


def find_l2c_file(path: Path) -> Path:
    """Resolve the L2C file to plot from a file or a SAFE output directory.

    Args:
        path: A single L2C ``.nc`` file, or a directory containing them. When a
            directory is given, the first (lowest subswath) L2C file is used.

    Returns:
        The resolved L2C file path

    Raises:
        FileNotFoundError: If no L2C file can be located
    """
    if path.is_file():
        return path
    files = sorted(path.glob("s1a-*_v*.nc")) or sorted(path.glob("*.nc"))
    if not files:
        msg = f"No L2C .nc file found in {path}"
        raise FileNotFoundError(msg)
    logger.info("Using L2C file %s from %s", files[0].name, path)
    return files[0]


def find_ww3_group(l2c_file: Path) -> str:
    """Return the name of the WW3 group present in the L2C file.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        The WW3 group name (``WW3`` or ``WW3_{intraburst,interburst}``)

    Raises:
        KeyError: If no WW3 group is found
    """
    with netCDF4.Dataset(l2c_file) as nc:  # pylint: disable=no-member
        groups = set(nc.groups)
    for group in WW3_GROUP_CANDIDATES:
        if group in groups:
            return group
    msg = f"No WW3 group found in {l2c_file} (tried {WW3_GROUP_CANDIDATES})"
    raise KeyError(msg)


def open_ww3_group(l2c_file: Path) -> xr.Dataset:
    """Open the WW3 group from the L2C file, whichever mode produced it.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        The WW3 dataset
    """
    return xr.open_dataset(l2c_file, group=find_ww3_group(l2c_file))


def sar_footprint(l2c_file: Path) -> MultiPoint:
    """Build the SAR subswath footprint from the tile corner coordinates.

    Args:
        l2c_file: Path to the L2C file

    Returns:
        A MultiPoint of all tile corner coordinates (may be empty)
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


def make_plot(
    l2c_file: Path,
    variable: str,
    output_png: Path,
) -> None:
    """Create the geographic map of a WW3 integrated parameter.

    Args:
        l2c_file: Path to the L2C file
        variable: WW3 integrated-parameter variable name to plot (e.g. ``hs``)
        output_png: Destination PNG path
    """
    ww3 = open_ww3_group(l2c_file)
    try:
        if variable not in ww3.data_vars:
            available = [v for v in ww3.data_vars if v not in ("longitude", "latitude")]
            msg = f"Variable {variable!r} not in the WW3 group (available: {available})"
            raise KeyError(msg)

        lons = ww3["longitude"].to_numpy().astype("f8")
        lats = ww3["latitude"].to_numpy().astype("f8")
        values = ww3[variable].to_numpy().astype("f4")
        units = ww3[variable].attrs.get("units", "")
        long_name = ww3[variable].attrs.get("long_name", variable)
    finally:
        ww3.close()

    valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(values)
    lons, lats, values = lons[valid], lats[valid], values[valid]
    if lons.size == 0:
        msg = f"No finite {variable} values to plot in {l2c_file}"
        raise ValueError(msg)

    footprint = sar_footprint(l2c_file)
    hull = footprint.convex_hull if footprint.geoms else None

    # Determine the map extent from the WW3 points (with a margin)
    pad_lon = max(1.0, 0.1 * (float(lons.max()) - float(lons.min())))
    pad_lat = max(1.0, 0.1 * (float(lats.max()) - float(lats.min())))
    left = float(lons.min()) - pad_lon
    right = float(lons.max()) + pad_lon
    bottom = float(lats.min()) - pad_lat
    top = float(lats.max()) + pad_lat

    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([left, right, bottom, top], crs=ccrs.PlateCarree())

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

    # SAR subswath footprint
    if hull is not None and not hull.is_empty:
        xs, ys = hull.exterior.xy
        ax.plot(
            xs,
            ys,
            color="#d62728",
            linewidth=1.6,
            linestyle="--",
            label="SAR subswath footprint",
        )

    # WW3 track points colored by the variable
    scatter = ax.scatter(
        lons,
        lats,
        c=values,
        cmap="viridis",
        s=28,
        edgecolor="k",
        linewidth=0.4,
        transform=ccrs.PlateCarree(),
        label=f"WW3 track points ({lons.size})",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(f"{variable} [{units}]")

    title = (
        f"WW3 {long_name}\nsampled at {lons.size} WW3 track points "
        f"({Path(l2c_file).name})"
    )
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_png)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Plot a WW3 integrated-parameter map from an L2C file"
    )
    parser.add_argument(
        "--input-l2c-safe",
        required=True,
        help="Path to a single L2C .nc file or a directory containing L2C files",
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

    l2c_file = find_l2c_file(Path(args.input_l2c_safe))
    make_plot(l2c_file, args.variable, Path(args.output_png))


if __name__ == "__main__":
    main()
