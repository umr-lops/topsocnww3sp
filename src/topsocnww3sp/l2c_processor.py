#!/usr/bin/env python3
"""L2C processor for Sentinel-1 OSW and WW3 colocalization."""

import argparse
import importlib.metadata
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely import contains_xy
from shapely.geometry import MultiPoint

from topsocnww3sp.read_s1_osw_tops_data import read_osw
from topsocnww3sp.utils import (
    gnomonic_projection,
    haversine,
    load_ww3_field,
    load_ww3_multi_grid,
    spherical_centroid,
    tidy_ww3_track_dataset,
)

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
    found_files = sorted(Path(base_dir).rglob(f"*_{year_month}_trck.nc"))
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
        error_msg = f"Measurement directory not found: {measurement_dir}"
        raise FileNotFoundError(error_msg)

    osw_files = list(measurement_dir.glob("*osw*.nc"))

    if not osw_files:
        error_msg = f"No OSW .nc files found in {measurement_dir}"
        raise FileNotFoundError(error_msg)

    # Sort to ensure consistent order (IW1, IW2, IW3 or EW1, EW2, ...)
    osw_files.sort()

    logger.info(
        "Found %d OSW files in SAFE: %s", len(osw_files), [f.name for f in osw_files]
    )
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

    # Flatten the tiles MultiIndex to a single dim. Each OSW file is a single
    # subswath, so no extra subswath dimension is needed (issue #22).
    ds_reset = fat_osw.reset_index("tiles")
    ds_stacked = ds_reset.rename({"tiles": "all_tiles"})
    valid_mask = ~np.isnan(ds_stacked["oswLon"].to_numpy())
    sar_flat = ds_stacked.isel(all_tiles=valid_mask)
    sar_flat = sar_flat.assign_coords(all_tiles=np.arange(len(sar_flat.all_tiles)))
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

    ds_sar = sar_flat.assign_coords(all_tiles=tile_coords)

    return ds_sar, ds_ww3_out, ds_match


def normalize_corner_arrays(
    lon_corners: np.ndarray,
    lat_corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Normalize SAR tile-corner arrays to shape (n_tiles, 4).

    Accepts layouts (tiles, corner), (corner, tiles) and higher-dim
    (..., corner) / (corner, ...) (issue #22: no subswath dimension).

    Args:
        lon_corners: Tile corner longitudes.
        lat_corners: Tile corner latitudes (same shape as lon_corners).

    Returns:
        (lon_corners, lat_corners) each of shape (n_tiles, 4), or None if
        the layout is not recognized.
    """
    if lon_corners.ndim > 2:
        if lon_corners.shape[-1] == 4:
            lon_corners = lon_corners.reshape(-1, 4)
            lat_corners = lat_corners.reshape(-1, 4)
        elif lon_corners.shape[0] == 4:
            lon_corners = lon_corners.reshape(4, -1).T
            lat_corners = lat_corners.reshape(4, -1).T
        else:
            logger.error("Unexpected corner array shape: %s", lon_corners.shape)
            return None
    elif lon_corners.ndim == 2 and lon_corners.shape[1] == 4:
        pass
    elif lon_corners.ndim == 2 and lon_corners.shape[0] == 4:
        lon_corners = lon_corners.T
        lat_corners = lat_corners.T
    else:
        logger.error("Unexpected corner array shape: %s", lon_corners.shape)
        return None
    return lon_corners, lat_corners


def process_lasso_group(
    osw_path: Path,
    ds_ww3: xr.Dataset,
    group_name: str,
    config: dict[str, Any],
    sar_start: datetime,
    fat_osw: xr.Dataset | None = None,
) -> xr.Dataset | None:
    """Lasso mode: extract WW3 points within buffered footprint of the SAR subswath.

    Args:
        osw_path: Path to the OSW file (subswath)
        ds_ww3: WW3 track dataset (``time`` dimension)
        group_name: SAR group used to build the footprint (e.g. "intraburst")
        config: Configuration dictionary with thresholds
        sar_start: Start time of SAR acquisition
        fat_osw: SAR data of ``group_name`` already read from ``osw_path``;
            read from the file when not provided

    Returns:
        WW3 spectra inside the buffered footprint on a ``sample`` dimension,
        or None when there is no match.
    """
    logger.info("--- Lasso Mode: %s ---", group_name)

    # 1. Load SAR data (unless the caller already did)
    if fat_osw is None:
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

    # 3. Extract corners and normalize to (n_tiles, 4)
    lon_corners = fat_osw["oswLongitudeCorner"].to_numpy()
    lat_corners = fat_osw["oswLatitudeCorner"].to_numpy()
    logger.debug("Corner array shape:  %s", lon_corners.shape)

    normalized = normalize_corner_arrays(lon_corners, lat_corners)
    if normalized is None:
        return None
    lon_corners, lat_corners = normalized

    corner_lons = lon_corners.astype("f8").ravel()
    corner_lats = lat_corners.astype("f8").ravel()
    valid = np.isfinite(corner_lons) & np.isfinite(corner_lats)
    corner_lons = corner_lons[valid]
    corner_lats = corner_lats[valid]
    if corner_lons.size < 3:
        logger.warning("Less than 3 corner points, cannot build polygon")
        return None

    # Build the footprint in a gnomonic projection centred on the subswath, so
    # that it stays compact across the +/-180 meridian and near the poles (a
    # hull in raw lon/lat spans the whole globe there). The buffer is applied
    # in the same plane: BUFFER_DEG of great-circle distance ~ tan(BUFFER_DEG).
    lon0, lat0 = spherical_centroid(corner_lons, corner_lats)
    cx, cy = gnomonic_projection(corner_lons, corner_lats, lon0, lat0)
    hull = MultiPoint(np.column_stack((cx, cy))).convex_hull
    buffered_polygon = hull.buffer(float(np.tan(np.radians(buffer_deg))))

    # Find WW3 points inside the polygon (vectorised; points on the far
    # hemisphere project to NaN and are never inside)
    wx, wy = gnomonic_projection(
        ww3_lons.astype("f8"), ww3_lats.astype("f8"), lon0, lat0
    )
    spatial_mask = np.asarray(contains_xy(buffered_polygon, wx, wy), dtype=bool)

    if not np.any(spatial_mask):
        logger.info("No WW3 points inside the buffered subswath footprint")
        return None

    ds_ww3_filtered = tidy_ww3_track_dataset(ds_ww3_subset.isel(time=spatial_mask))

    # WW3 spectra are individual observations, not a time series: rename the
    # "time" dimension to "sample" and keep the observation times as a regular
    # "time" variable (issue #23). The CF time-encoding attrs are dropped so the
    # variable is written as a plain datetime data variable, not a coordinate.
    obs_time = ds_ww3_filtered["time"].rename({"time": "sample"}).copy()
    obs_time.attrs = {"long_name": "WW3 observation time (julian day, UT)"}
    ds_ww3_filtered = ds_ww3_filtered.rename({"time": "sample"})
    ds_ww3_filtered["time"] = obs_time

    ds_ww3_filtered.attrs["sar_file"] = str(osw_path.name)
    ds_ww3_filtered.attrs["buffer_deg"] = buffer_deg

    logger.info("Selected %d WW3 points out of %d", np.sum(spatial_mask), len(ww3_lons))
    return ds_ww3_filtered


# SAFE product type, e.g. "OCN__2SDV": level 2, standard class, dual VV+VH
_SAFE_POLARIZATION_RE = re.compile(r"_\d[SA](SH|SV|DH|DV|HH|VV|HV|VH)_")


def read_osw_root_attrs(osw_path: Path) -> dict[str, Any]:
    """Read the global (root group) attributes of an ESA OSW file.

    They hold the IPF version, the source SLC product, the OSW algorithm
    version, etc., which are not visible from the intraburst/interburst groups.

    Args:
        osw_path: Path to the OSW .nc file

    Returns:
        Dict of root attributes (empty if the file cannot be read)
    """
    try:
        with xr.open_dataset(osw_path) as ds_root:
            return dict(ds_root.attrs)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Cannot read global attributes of %s: %s", osw_path, exc)
        return {}


def sar_polarization_from_safe(safe_name: str) -> str | None:
    """Polarization code of a SAFE product name (``SH``, ``SV``, ``DH``, ``DV``).

    Args:
        safe_name: SAFE directory name, e.g.
            ``S1A_IW_OCN__2SDV_20220107T062432_..._F9CD.SAFE``

    Returns:
        The polarization code, or None if the name does not match
    """
    match = _SAFE_POLARIZATION_RE.search(safe_name)
    return match.group(1) if match else None


def create_global_attributes(
    sar_filename: str,
    product_version: str,
    mode: str,
    ww3_field_vars: list[str] | None = None,
    safe_name: str | None = None,
    osw_attrs: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Create CF-compliant global attributes for the L2C product.

    Args:
        sar_filename: Name of the SAR OSW file
        product_version: Product version string (e.g. ``v0.2``)
        mode: Processing mode
        ww3_field_vars: Optional list of WW3 FIELD (integrated parameter)
            variable names included in the product, to note in the comment
        safe_name: Name of the input OCN SAFE directory
        osw_attrs: Global attributes of the input OSW file (IPF version...)

    Returns:
        Dictionary of global attributes
    """
    try:
        pkg_version = importlib.metadata.version("topsocnww3sp")
    except importlib.metadata.PackageNotFoundError:
        pkg_version = "unknown"
    osw_attrs = osw_attrs or {}

    # OSW file name: {mission}-{swath}-osw-{pol}-{start}-{stop}-{orbit}-{take}-{index}.nc
    stem = Path(sar_filename).stem
    parts = stem.split("-")
    mission = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
    subswath = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
    osw_polarization = parts[3].upper() if len(parts) > 3 else "UNKNOWN"
    osw_polarization = str(osw_attrs.get("polarisation", osw_polarization)).upper()
    sar_polarization = (
        sar_polarization_from_safe(safe_name) if safe_name else None
    ) or osw_polarization
    now_str = datetime.now(tz=timezone.utc).isoformat()
    product_id = f"{stem}_{product_version}"

    field_note = ""
    if ww3_field_vars:
        field_note = " WW3 integrated parameters from FIELD files: " + ", ".join(
            ww3_field_vars
        )

    attrs = {
        "Conventions": "CF-1.8, ACDD-1.3",
        "title": (
            f"Sentinel-1 {mission} {subswath} {osw_polarization} OSW and WW3 "
            "spectra L2C co-location product"
        ),
        "summary": (
            f"Sentinel-1 {mission} {subswath} {osw_polarization} OSW (Ocean "
            f"Swell spectra) data co-located with WW3 spectra using {mode} mode"
        ),
        "keywords": "SAR, Sentinel-1, WW3, ocean waves, spectra",
        "keywords_vocabulary": "",
        "history": f"{now_str} - Created by topsocnww3sp version {pkg_version}",
        "source": f"Sentinel-1 {mission} OSW product: {Path(sar_filename).name}",
        "references": "https://github.com/umr-lops/topsocnww3sp",
        "comment": f"Co-localization mode: {mode}.{field_note}",
        "license": "CC-BY-4.0",
        "id": product_id,
        "naming_authority": "fr.ifremer.lops",
        "product_id": product_id,
        "product_id_description": (
            "Unique identifier of this L2C file: the name of the input ESA OSW "
            "file (mission-subswath-osw-polarization-start-stop-absolute orbit-"
            "mission data take-image index) followed by the L2C product version"
        ),
        "product_version": product_version,
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
        "sar_polarization": sar_polarization,
        "sar_osw_polarization": osw_polarization,
        "sar_ipf_version": str(osw_attrs.get("IPFversion", "unknown")),
        "creator_name": "Antoine Grouazel",
        "creator_email": "antoine.grouazel@ifremer.fr",
        "creator_institution": "Ifremer",
        "institution": "Ifremer",
        "project": "TOPSOCNWW3",
        "publisher_name": "Ifremer/LOPS-SIAM",
        "publisher_email": "antoine.grouazel@ifremer.fr",
        "publisher_institution": "Ifremer",
    }
    if safe_name:
        attrs["sar_safe"] = safe_name
    for key, osw_key in (
        ("sar_source_product", "sourceProduct"),
        ("sar_osw_algorithm_version", "oswAlgorithmVersion"),
        ("sar_processing_center", "processingCenter"),
    ):
        if osw_key in osw_attrs:
            attrs[key] = str(osw_attrs[osw_key])
    return attrs


# NetCDF zlib compression for the L2C output (issue #25). Hardcoded: only the
# large spectral arrays are compressed at level 4. Small per-tile / coordinate
# arrays are left uncompressed because the per-chunk zlib overhead would make
# them bigger than their raw size.
_COMPRESSION_LEVEL = 4
_COMPRESSION_MIN_BYTES = 4096


def _compression_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Build a NetCDF zlib encoding dict for the large arrays of ``ds``.

    Numeric variables whose in-memory size is at least ``_COMPRESSION_MIN_BYTES``
    are written with ``zlib`` at level ``_COMPRESSION_LEVEL``; the rest (small
    per-tile vectors, coordinates, flags) are left uncompressed.

    Args:
        ds: Dataset to derive the per-variable encoding from.

    Returns:
        Mapping of variable name to its NetCDF encoding options.
    """
    encoding: dict[str, dict[str, Any]] = {}
    for name, var in ds.variables.items():
        if var.dtype.kind not in "iuf":
            continue
        nbytes = int(var.size * var.dtype.itemsize)
        if nbytes >= _COMPRESSION_MIN_BYTES:
            encoding[str(name)] = {"zlib": True, "complevel": _COMPRESSION_LEVEL}
    return encoding


def write_output_file(
    output_path: Path,
    global_attrs: dict[str, str],
    sar_groups: list[tuple[str, xr.Dataset]],
    ww3_groups: list[tuple[str, xr.Dataset]],
    match_groups: list[tuple[str, xr.Dataset]] | None = None,
) -> None:
    """Write a complete L2C output file with global attributes and groups."""
    # Create the output directory only when a file is actually written, so no
    # empty SAFE directory is left behind when no colocation data is extracted.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize file with global attributes
    xr.Dataset(attrs=global_attrs).to_netcdf(output_path, mode="w")

    # Write SAR groups
    for group_name, ds in sar_groups:
        ds.to_netcdf(
            output_path,
            group=group_name,
            mode="a",
            encoding=_compression_encoding(ds),
        )

    # Write WW3 groups
    for group_name, ds in ww3_groups:
        ds_copy = ds.copy()
        ds_copy.encoding.clear()
        ds_copy.to_netcdf(
            output_path,
            group=group_name,
            mode="a",
            encoding=_compression_encoding(ds_copy),
        )

    # Write MATCH_MAP groups if provided
    if match_groups:
        for group_name, ds in match_groups:
            ds.to_netcdf(
                output_path,
                group=group_name,
                mode="a",
                encoding=_compression_encoding(ds),
            )


def load_ww3(
    ww3_path: Path, sar_start: datetime, config: dict[str, Any]
) -> tuple[xr.Dataset, list[str]]:
    """Load WW3 spectra (single file or multi-grid run dir) and FIELD variables.

    Args:
        ww3_path: WW3 track file, or WW3 run directory (multi-grid mode)
        sar_start: SAR acquisition start time
        config: Configuration dictionary

    Returns:
        (WW3 track dataset, sorted list of FIELD variable names added)

    Raises:
        FileNotFoundError, ValueError: when the WW3 data cannot be loaded
    """
    if ww3_path.is_dir():
        logger.info("Multi-grid mode: loading WW3 data from directory %s", ww3_path)
        ds_ww3 = load_ww3_multi_grid(ww3_path, sar_start, config)
    else:
        logger.info("Single-file mode: loading WW3 data from %s", ww3_path)
        ds_ww3 = xr.open_dataset(ww3_path)
    ds_ww3 = tidy_ww3_track_dataset(ds_ww3)

    # Sample WW3 FIELD (integrated parameter) variables at the track points so
    # that each spectrum carries its own integrated parameters on the same
    # dimension (issue #20). Never fatal: the product is built without them.
    ww3_field_vars: list[str] = []
    try:
        field_vars = load_ww3_field(ww3_path, ds_ww3, config)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "Failed to load WW3 FIELD variables, continuing without them: %s", exc
        )
        field_vars = {}
    if field_vars:
        ds_ww3 = ds_ww3.copy()
        for field_name, field_var in field_vars.items():
            ds_ww3[field_name] = field_var
        ww3_field_vars = sorted(field_vars)
        logger.info("Loaded WW3 FIELD variables: %s", ", ".join(ww3_field_vars))
    return ds_ww3, ww3_field_vars


def sar_start_from_osw_name(osw_file: Path) -> datetime:
    """SAR acquisition start time (UTC) from an OSW file name."""
    return datetime.strptime(osw_file.name.split("-")[4], "%Y%m%dt%H%M%S").replace(
        tzinfo=timezone.utc
    )


def process_safe(
    safe_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    mode: str = "lasso",
    ww3_path: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Build the L2C products (one per subswath) of one OCN SAFE.

    Args:
        safe_path: OCN SAFE directory
        config: Configuration dictionary
        output_dir: Base output directory (``YYYY/MM/DD/<SAFE>/`` is appended)
        mode: Matching mode (lasso, 1to1, unique, many)
        ww3_path: WW3 track file or run directory; searched from the config
            ``directory_ww3spectra_output`` when None
        overwrite: Overwrite existing output files

    Returns:
        List of the L2C files written (empty when there is no co-location or
        every output already exists)

    Raises:
        FileNotFoundError, ValueError: missing OSW files or WW3 data
    """
    osw_files = list_osw_files_in_safe(safe_path)
    sar_start = sar_start_from_osw_name(osw_files[0])

    product_version = config.get("product_version", "v0.1")
    safe_name = safe_path.name
    # The output directory is created lazily by write_output_file, only when a
    # colocation product is actually written (issue #15).
    output_base_dir = (
        output_dir
        / sar_start.strftime("%Y")
        / sar_start.strftime("%m")
        / sar_start.strftime("%d")
        / safe_name
    )

    todo: list[tuple[Path, Path]] = []
    for osw_file in osw_files:
        output_path = output_base_dir / f"{osw_file.stem}_{product_version}.nc"
        if output_path.exists() and not overwrite:
            logger.info("Output file %s already exists. Skipping...", output_path)
            continue
        todo.append((osw_file, output_path))
    if not todo:
        return []

    if ww3_path is None:
        ww3_path = Path(find_ww3_file(sar_start, config))
    ds_ww3, ww3_field_vars = load_ww3(ww3_path, sar_start, config)

    groups = ["intraburst", "interburst"]
    written: list[Path] = []

    # Process each OSW file (one per subswath)
    for osw_file, output_path in todo:
        subswath_name = osw_file.stem.split("-")[1]  # iw1, iw2, iw3, etc.
        logger.info("=" * 60)
        logger.info("Processing subswath: %s", subswath_name)
        logger.info("=" * 60)

        if output_path.exists() and overwrite:
            output_path.unlink()

        global_attrs = create_global_attributes(
            osw_file.name,
            product_version,
            mode,
            ww3_field_vars,
            safe_name=safe_name,
            osw_attrs=read_osw_root_attrs(osw_file),
        )

        # SAR groups keep the names of the ESA OSW groups (issue #21)
        sar_groups: list[tuple[str, xr.Dataset]] = []
        ww3_groups: list[tuple[str, xr.Dataset]] = []
        match_groups: list[tuple[str, xr.Dataset]] = []

        if mode == "lasso":
            sar_data: dict[str, xr.Dataset] = {}
            for g in groups:
                fat_osw, _ = read_osw(g, [osw_file])
                if fat_osw is None:
                    continue
                sar_data[g] = fat_osw
                if "tiles" in fat_osw.dims or "tiles" in fat_osw.coords:
                    sar_groups.append((g, fat_osw.reset_index("tiles")))
                else:
                    sar_groups.append((g, fat_osw))

            # WW3 footprint from the intraburst tiles (read once above)
            if "intraburst" in sar_data:
                ds_ww3_selected = process_lasso_group(
                    osw_file,
                    ds_ww3,
                    "intraburst",
                    config,
                    sar_start,
                    fat_osw=sar_data["intraburst"],
                )
                if ds_ww3_selected is not None:
                    ww3_groups.append(("WW3", ds_ww3_selected))
        else:  # modes 1to1, unique, many
            for g in groups:
                d_sar, d_ww3, d_match = process_group(
                    osw_file, ds_ww3, g, config, sar_start, mode
                )
                if d_sar is not None and d_ww3 is not None and d_match is not None:
                    sar_groups.append((g, d_sar))
                    ww3_groups.append((f"WW3_{g}", d_ww3))
                    match_groups.append((f"MATCH_MAP_{g}", d_match))

        # A product needs SAR data and co-located WW3 spectra
        if sar_groups and ww3_groups:
            write_output_file(
                output_path, global_attrs, sar_groups, ww3_groups, match_groups or None
            )
            logger.info("Successfully wrote %s", output_path)
            written.append(output_path)
        else:
            logger.warning("No co-location for %s", subswath_name)

    return written


def main() -> None:
    """Main function to process SAR and WW3 data for colocalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ocn-safe",
        required=True,
        help="path to SAFE directory containing OSW .nc files (IW1, IW2, IW3 or EW1..EW5)",
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

    try:
        process_safe(
            Path(args.ocn_safe),
            config,
            Path(args.output_dir),
            mode=args.mode,
            ww3_path=Path(args.ww3_file) if args.ww3_file else None,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError):
        logger.exception("Failed to process %s", args.ocn_safe)
        raise SystemExit(1) from None

    logger.info("Done.")


if __name__ == "__main__":
    main()
