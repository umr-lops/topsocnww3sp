#!/usr/bin/env python3
"""L2C processor for Sentinel-1 OSW and WW3 colocalization."""

import argparse
import importlib.metadata
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely.geometry import MultiPoint, Point

from topsocnww3sp.read_s1_osw_tops_data import read_osw
from topsocnww3sp.utils import haversine, load_ww3_multi_grid

logger = logging.getLogger(__name__)


def find_ww3_file(sar_time: datetime, config: dict[str, Any]) -> str:
    """Find the WW3 file corresponding to the SAR acquisition time.

    Args:
        sar_time: SAR acquisition time
        config: Configuration dictionary containing directory_ww3spectra_output

    Returns:
        Path to the WW3 file

    Raises:
        FileNotFoundError: If no WW3 file is found for the given time period
    """
    base_dir = config["directory_ww3spectra_output"]
    year_month = sar_time.strftime("%Y%m")
    search_pattern = Path(base_dir) / "**" / f"*_{year_month}_trck.nc"
    found_files = list(search_pattern.rglob("*"))
    if not found_files:
        msg = f"No WW3 file for {year_month} in {base_dir}"
        raise FileNotFoundError(msg)
    return str(found_files[0])


def list_osw_files_in_safe(safe_path: Path) -> list[Path]:
    """List all OSW .nc files in a SAFE directory.

    Args:
        safe_path: Path to the SAFE directory

    Returns:
        List of paths to OSW .nc files (one per subswath)

    Raises:
        FileNotFoundError: If measurement directory or OSW files not found
    """
    measurement_dir = safe_path / "measurement"
    if not measurement_dir.exists():
        raise FileNotFoundError(f"Measurement directory not found: {measurement_dir}")

    osw_files = list(measurement_dir.glob("*osw*.nc"))

    if not osw_files:
        raise FileNotFoundError(f"No OSW .nc files found in {measurement_dir}")

    # Sort to ensure consistent order (IW1, IW2, IW3 or EW1, EW2, ...)
    osw_files.sort()

    logger.info("Found %d OSW files in SAFE: %s", len(osw_files), [f.name for f in osw_files])
    return osw_files


def process_group(
    osw_path: Path,
    ds_ww3: xr.Dataset,
    group_name: str,
    config: dict[str, Any],
    sar_start: datetime,
    mode: str,
) -> tuple[xr.Dataset | None, xr.Dataset | None, xr.Dataset | None]:
    """Process a specific group (intraburst or interburst) of SAR data.

    Args:
        osw_path: Path to the OSW file
        ds_ww3: WW3 dataset containing spectral data
        group_name: Name of the group to process ("intraburst" or "interburst")
        config: Configuration dictionary with thresholds
        sar_start: Start time of SAR acquisition
        mode: Processing mode ("1to1", "unique", or "many")

    Returns:
        Tuple containing (ds_sar, ds_ww3_out, ds_match) or (None, None, None) if no matches
    """
    logger.info("--- Processing Group: %s [Mode: %s] ---", group_name, mode)

    # 1. Load and Flatten SAR
    fat_osw, _ = read_osw(group_name, [osw_path])
    if fat_osw is None or len(fat_osw.data_vars) == 0:
        return None, None, None

    # fat_osw["time"] = sar_start
    fat_osw["time"] = np.datetime64(sar_start.replace(tzinfo=None), "ns")

    # First reset the existing MultiIndex on tiles, then stack all spatial dims flat
    ds_reset = fat_osw.reset_index("tiles")
    # PD013: stack is fine here, we keep it
    ds_stacked = ds_reset.stack(all_tiles=("subswath", "tiles"))  # noqa: PD013
    valid_mask = ~np.isnan(ds_stacked["oswLon"].to_numpy())
    sar_flat = ds_stacked.isel(all_tiles=valid_mask)
    sar_flat = sar_flat.reset_index("tiles", drop=True)
    if "subswath" in sar_flat.dims:
        sar_flat = sar_flat.drop_vars("subswath")
    if "all_tiles" in sar_flat.dims:
        sar_flat = sar_flat.reset_index("all_tiles").assign_coords(
            all_tiles=np.arange(len(sar_flat.all_tiles))
        )
    n_tiles = len(sar_flat.all_tiles)

    # 2. Temporal Filter WW3
    ww3_times = pd.to_datetime(ds_ww3.time.to_numpy())
    t_sar = pd.Timestamp(sar_start).tz_localize(None)
    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)

    if not np.any(t_mask):
        return None, None, None

    cand_idx_orig = np.where(t_mask)[0]
    cand_lons: np.ndarray = ds_ww3.longitude.to_numpy()[t_mask]
    cand_lats: np.ndarray = ds_ww3.latitude.to_numpy()[t_mask]
    dist_thresh = config["DISTANCE_THRESHOLD_KM"]

    # 3. Matchup Logic
    sar_indices: list[int] = []
    ww3_indices_rel: list[int] = []
    distances: list[float] = []

    for i in range(n_tiles):
        tile = sar_flat.isel(all_tiles=i)
        lon_val = float(tile.oswLon)
        lat_val = float(tile.oswLat)
        dists: np.ndarray = haversine(lon_val, lat_val, cand_lons, cand_lats)

        if mode in ["1to1", "unique"]:
            min_idx: int = int(np.argmin(dists))
            if dists[min_idx] <= dist_thresh:
                sar_indices.append(i)
                ww3_indices_rel.append(min_idx)
                distances.append(dists[min_idx])
        elif mode == "many":
            matches = np.where(dists <= dist_thresh)[0]
            for m in matches:
                sar_indices.append(i)
                ww3_indices_rel.append(m)
                distances.append(dists[m])

    if not sar_indices:
        return None, None, None

    tile_coords = np.arange(n_tiles)

    if mode == "1to1":
        ww3_ptr = np.full(n_tiles, -1, dtype=int)
        ww3_ptr[sar_indices] = cand_idx_orig[ww3_indices_rel]

        ds_ww3_sel = ds_ww3.isel(time=ww3_ptr)
        actual_ww3_times = ds_ww3_sel.time.to_numpy()
        ds_ww3_out = ds_ww3_sel.rename({"time": "all_tiles"})
        ds_ww3_out = ds_ww3_out.assign_coords(all_tiles=tile_coords)
        ds_ww3_out["time"] = (["all_tiles"], actual_ww3_times)

        ww3_dt = pd.to_datetime(actual_ww3_times)
        t_diffs = (ww3_dt - t_sar).total_seconds().to_numpy()

        ds_match = xr.Dataset(
            {
                "distance_km": (
                    ["all_tiles"],
                    pd.Series(distances, index=sar_indices)
                    .reindex(tile_coords)
                    .to_numpy(),
                ),
                "time_diff_sec": (["all_tiles"], t_diffs),
            },
            coords={"all_tiles": tile_coords},
        )

    elif mode == "unique":
        unique_rel_idx, inverse_map = np.unique(ww3_indices_rel, return_inverse=True)
        ds_ww3_out = ds_ww3.isel(time=cand_idx_orig[unique_rel_idx]).rename(
            {"time": "unique_ww3"}
        )

        ptr = np.full(n_tiles, -1, dtype=int)
        ptr[sar_indices] = inverse_map

        ds_match = xr.Dataset(
            {
                "ww3_ptr": (["all_tiles"], ptr),
                "distance_km": (
                    ["all_tiles"],
                    pd.Series(distances, index=sar_indices)
                    .reindex(tile_coords)
                    .to_numpy(),
                ),
            },
            coords={"all_tiles": tile_coords},
        )

    elif mode == "many":
        unique_rel_idx, inverse_map = np.unique(ww3_indices_rel, return_inverse=True)
        ds_ww3_out = ds_ww3.isel(time=cand_idx_orig[unique_rel_idx]).rename(
            {"time": "unique_ww3"}
        )

        ds_match = xr.Dataset(
            {
                "sar_index": (["pair"], sar_indices),
                "ww3_index": (["pair"], inverse_map),
                "distance_km": (["pair"], distances),
            },
            coords={"pair": np.arange(len(sar_indices))},
        )
    else:
        return None, None, None

    ds_sar = sar_flat.reset_index("all_tiles").assign_coords(all_tiles=tile_coords)

    return ds_sar, ds_ww3_out, ds_match


def process_lasso_group(
    osw_path: Path,
    ds_ww3: xr.Dataset,
    group_name: str,
    config: dict[str, Any],
    sar_start: datetime,
) -> xr.Dataset | None:
    """Lasso mode: extract WW3 points within buffered footprint of the SAR subswath."""
    logger.info("--- Lasso Mode: %s ---", group_name)

    # 1. Load SAR data
    fat_osw, _ = read_osw(group_name, [osw_path])
    if fat_osw is None or len(fat_osw.data_vars) == 0:
        return None

    # 2. Temporal filter on WW3
    ww3_times = pd.to_datetime(ds_ww3.time.to_numpy())
    t_sar = pd.Timestamp(sar_start).tz_localize(None)
    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)
    if not np.any(t_mask):
        logger.info("No WW3 data within time window")
        return None

    ds_ww3_subset = ds_ww3.isel(time=t_mask)
    ww3_lons = ds_ww3_subset.longitude.to_numpy()
    ww3_lats = ds_ww3_subset.latitude.to_numpy()
    buffer_deg = config.get("BUFFER_DEG", 0.1)

    # 3. Extract corners - handle shape (subswath, corner, tiles)
    lon_corners = fat_osw["oswLongitudeCorner"].to_numpy()
    lat_corners = fat_osw["oswLatitudeCorner"].to_numpy()

    logger.debug("Corner array shape:  %s", lon_corners.shape)

    # Handle different possible structures
    if lon_corners.ndim == 3:
        if lon_corners.shape[0] == 1 and lon_corners.shape[1] == 4:
            lon_corners = lon_corners[0]
            lat_corners = lat_corners[0]

        if lon_corners.ndim == 2 and lon_corners.shape[0] == 4:
            lon_corners = lon_corners.T
            lat_corners = lat_corners.T

    elif lon_corners.ndim == 4:
        lon_corners = lon_corners.reshape(-1, 4)
        lat_corners = lat_corners.reshape(-1, 4)

    elif lon_corners.ndim == 2 and lon_corners.shape[1] == 4:
        pass
    else:
        logger.error("Unexpected corner array shape: %s", lon_corners.shape)
        return None

    # Build all corners
    all_corners = [
        (float(lon_corners[i, k]), float(lat_corners[i, k]))
        for i in range(lon_corners.shape[0])
        for k in range(4)
    ]

    if len(all_corners) < 3:
        logger.warning("Less than 3 corner points, cannot build polygon")
        return None

    points = [Point(lon, lat) for lon, lat in all_corners]
    multipoint = MultiPoint(points)
    hull = multipoint.convex_hull
    buffered_polygon = hull.buffer(buffer_deg)

    # Find WW3 points inside the polygon
    spatial_mask = np.array(
        [
            buffered_polygon.contains(Point(float(lon), float(lat)))
            for lon, lat in zip(ww3_lons, ww3_lats, strict=False)
        ]
    )

    if not np.any(spatial_mask):
        logger.info("No WW3 points inside the buffered subswath footprint")
        return None

    ds_ww3_filtered = ds_ww3_subset.isel(time=spatial_mask)

    ds_ww3_filtered.attrs["sar_file"] = str(osw_path.name)
    ds_ww3_filtered.attrs["buffer_deg"] = buffer_deg

    logger.info("Selected %d WW3 points out of %d", np.sum(spatial_mask), len(ww3_lons))
    return ds_ww3_filtered


def create_global_attributes(
    sar_filename: str,
    product_version: str,
    mode: str,
) -> dict[str, str]:
    """Create CF-compliant global attributes for the L2C product."""
    try:
        pkg_version = importlib.metadata.version("topsocnww3sp")
    except importlib.metadata.PackageNotFoundError:
        pkg_version = "unknown"

    parts = Path(sar_filename).stem.split("-")
    mission = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
    subswath = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
    polarization = parts[2].upper() if len(parts) > 2 else "UNKNOWN"
    now_str = datetime.now().isoformat()

    return {
        "Conventions": "CF-1.8",
        "title": f"S1 {subswath} {polarization} L2C Co-localization Product",
        "summary": (
            f"Sentinel-1 {mission} {subswath} {polarization} "
            f"OSW data co-localized with WW3 spectra using {mode} mode"
        ),
        "keywords": "SAR, Sentinel-1, WW3, ocean waves, spectra",
        "keywords_vocabulary": "",
        "history": f"{now_str} - Created by topsocnww3sp version {pkg_version}",
        "source": f"Sentinel-1 {mission} OSW product: {Path(sar_filename).name}",
        "references": "https://github.com/umr-lops/topsocnww3sp",
        "comment": f"Co-localization mode: {mode}",
        "license": "CC-BY-4.0",
        "product_id": product_version,
        "package_version": pkg_version,
        "provider": "Ifremer",
        "date_created": now_str,
        "date_modified": now_str,
        "processing_level": "Level-2C",
        "processing_facility": "Ifremer/LOPS-SIAM",
        "processor_name": "topsocnww3sp",
        "processor_version": pkg_version,
        "sar_mission": mission,
        "sar_subswath": subswath,
        "sar_polarization": polarization,
        "creator_name": "Antoine Grouazel",
        "creator_email": "antoine.grouazel@ifremer.fr",
        "creator_institution": "Ifremer",
        "institution": "Ifremer",
        "project": "TOPSOCNWW3",
        "publisher_name": "Ifremer/LOPS-SIAM",
        "publisher_email": "antoine.grouazel@ifremer.fr",
        "publisher_institution": "Ifremer",
    }


def write_output_file(
    output_path: Path,
    global_attrs: dict[str, str],
    sar_groups: list[tuple[str, xr.Dataset]],
    ww3_groups: list[tuple[str, xr.Dataset]],
    match_groups: list[tuple[str, xr.Dataset]] | None = None,
) -> None:
    """Write a complete L2C output file with global attributes and groups."""
    # Initialize file with global attributes
    xr.Dataset(attrs=global_attrs).to_netcdf(output_path, mode="w")

    # Write SAR groups
    for group_name, ds in sar_groups:
        ds.to_netcdf(output_path, group=group_name, mode="a")

    # Write WW3 groups
    for group_name, ds in ww3_groups:
        ds_copy = ds.copy()
        ds_copy.encoding.clear()
        ds_copy.to_netcdf(output_path, group=group_name, mode="a")

    # Write MATCH_MAP groups if provided
    if match_groups:
        for group_name, ds in match_groups:
            ds.to_netcdf(output_path, group=group_name, mode="a")


def main() -> None:
    """Main function to process SAR and WW3 data for colocalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocn-safe", required=True,
        help="path to SAFE directory containing OSW .nc files (IW1, IW2, IW3 or EW1..EW5)"
    )
    parser.add_argument(
        "--ww3-file",
        default=None,
        help="directory or path of nc file containing WW3 spectra. If not provided, the script will search based on SAR time and config directory",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file with thresholds and directories",
    )
    parser.add_argument(
        "--mode",
        choices=["1to1", "unique", "many", "lasso"],
        default="lasso",
        help="Matching mode: '1to1' (one WW3 per SAR), 'unique' (multiple SAR can share same WW3), or 'many' (all matches) or 'lasso' (all spectra within the footprint) [optional, default: lasso]",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output-dir", required=True, help="Base directory to save output files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files. [optional, default: False]",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.getLogger("topsocnww3sp.utils").setLevel(log_level)
    logging.getLogger("topsocnww3sp.read_s1_osw_tops_data").setLevel(log_level)

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    safe_path = Path(args.ocn_safe)
    osw_files = list_osw_files_in_safe(safe_path)

    # Extract SAR start time from first file (or from SAFE name if more robust)
    first_osw = osw_files[0]
    fname = first_osw.name
    parts = fname.split("-")
    sar_start = datetime.strptime(parts[4], "%Y%m%dt%H%M%S").replace(tzinfo=timezone.utc)

    product_version = config.get("product_version", "v0.1")
    safe_name = safe_path.name
    year = sar_start.strftime("%Y")
    month = sar_start.strftime("%m")
    day = sar_start.strftime("%d")
    output_base_dir = Path(args.output_dir) / year / month / day / safe_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load WW3 data once for all subswaths
    ww3_path = args.ww3_file or find_ww3_file(sar_start, config)
    ww3_path_obj = Path(ww3_path)

    if ww3_path_obj.is_dir():
        logger.info("Multi-grid mode: loading WW3 data from directory %s", ww3_path_obj)
        try:
            ds_ww3 = load_ww3_multi_grid(ww3_path_obj, sar_start, config)
        except (FileNotFoundError, ValueError):
            logger.exception("Failed to load WW3 multi-grid")
            return
    else:
        logger.info("Single-file mode: loading WW3 data from %s", ww3_path_obj)
        ds_ww3 = xr.open_dataset(ww3_path_obj)

    groups = ["intraburst", "interburst"]

    # Process each OSW file (one per subswath)
    for osw_file in osw_files:
        subswath_name = osw_file.stem.split("-")[1]  # iw1, iw2, iw3, etc.
        logger.info("=" * 60)
        logger.info("Processing subswath: %s", subswath_name)
        logger.info("=" * 60)

        output_name = f"{osw_file.stem}_{product_version}.nc"
        output_path = output_base_dir / output_name

        if output_path.exists() and not args.overwrite:
            logger.info("Output file %s already exists. Skipping...", output_path)
            continue
        if output_path.exists() and args.overwrite:
            output_path.unlink()

        global_attrs = create_global_attributes(osw_file.name, product_version, args.mode)

        if args.mode == "lasso":
            sar_groups = []
            ww3_groups = []

            # Process SAR groups
            for g in groups:
                fat_osw, _ = read_osw(g, [osw_file])
                if fat_osw is not None:
                    if "tiles" in fat_osw.dims or "tiles" in fat_osw.coords:
                        fat_osw_to_write = fat_osw.reset_index("tiles")
                    else:
                        fat_osw_to_write = fat_osw
                    sar_groups.append((f"SAR_{g}", fat_osw_to_write))

            # Process WW3 for this subswath
            ds_ww3_selected = process_lasso_group(
                osw_file, ds_ww3, "intraburst", config, sar_start
            )
            if ds_ww3_selected is not None:
                ww3_groups.append(("WW3", ds_ww3_selected))

            if sar_groups or ww3_groups:
                write_output_file(output_path, global_attrs, sar_groups, ww3_groups)
                logger.info("Successfully wrote %s", output_path)
            else:
                logger.warning("No data extracted for %s", subswath_name)

        else:  # modes 1to1, unique, many
            sar_groups = []
            ww3_groups = []
            match_groups = []

            for g in groups:
                result = process_group(osw_file, ds_ww3, g, config, sar_start, args.mode)
                d_sar, d_ww3, d_match = result
                if d_sar is not None and d_ww3 is not None and d_match is not None:
                    sar_groups.append((f"SAR_{g}", d_sar))
                    ww3_groups.append((f"WW3_{g}", d_ww3))
                    match_groups.append((f"MATCH_MAP_{g}", d_match))

            if sar_groups:
                write_output_file(output_path, global_attrs, sar_groups, ww3_groups, match_groups)
                logger.info("Successfully wrote %s", output_path)
            else:
                logger.warning("No data extracted for %s", subswath_name)

    logger.info("Done.")


if __name__ == "__main__":
    main()