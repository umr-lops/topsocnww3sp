#!/usr/bin/env python3
import argparse
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
from topsocnww3sp.utils import haversine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    search_pattern = Path(base_dir) / "**" / f"*_{year_month}_trck.nc"
    found_files = list(search_pattern.rglob("*"))
    if not found_files:
        msg = f"No WW3 file for {year_month} in {base_dir}"
        raise FileNotFoundError(msg)
    return str(found_files[0])


def process_group(
    osw_path: str,
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
    # read_osw is not yet typed, silence mypy
    fat_osw, _ = read_osw(group_name, [Path(osw_path)])
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
    # sar_flat = sar_flat.drop_vars(["all_tiles", "subswath", "tiles"]).assign_coords(
    #     all_tiles=np.arange(len(sar_flat.all_tiles))
    # )
    sar_flat = sar_flat.reset_index("tiles", drop=True)
    if "subswath" in sar_flat.dims:
        sar_flat = sar_flat.drop_vars("subswath")
    if "all_tiles" in sar_flat.dims:
        sar_flat = sar_flat.reset_index("all_tiles").assign_coords(
            all_tiles=np.arange(len(sar_flat.all_tiles))
        )
    # sar_flat = sar_flat.drop_vars(["all_tiles", "subswath"])
    n_tiles = len(sar_flat.all_tiles)

    # 2. Temporal Filter WW3
    ww3_times = pd.to_datetime(ds_ww3.time.to_numpy())
    t_sar = pd.Timestamp(sar_start).tz_localize(None)  # strip tzinfo for comparison
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
        # Extract scalar values from 0D arrays to satisfy mypy
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

        # 1. Select matched WW3 data
        ds_ww3_sel = ds_ww3.isel(time=ww3_ptr)
        # 2. Extract the actual WW3 timestamps before we rename the dimension
        actual_ww3_times = ds_ww3_sel.time.to_numpy()
        # 3. Rename dimension to align with SAR tiles
        ds_ww3_out = ds_ww3_sel.rename({"time": "all_tiles"})
        # 4. Re-assign simple integer coordinates to all_tiles
        ds_ww3_out = ds_ww3_out.assign_coords(all_tiles=tile_coords)
        # 5. Put the timestamps back into a variable named 'time'
        ds_ww3_out["time"] = (["all_tiles"], actual_ww3_times)

        # Calculate time diff
        # We handle NaTs (for non-matches) gracefully
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
        # Unreachable, but keep mypy happy
        return None, None, None

    # Ensure SAR group also has clean integer indices
    ds_sar = sar_flat.reset_index("all_tiles").assign_coords(all_tiles=tile_coords)

    return ds_sar, ds_ww3_out, ds_match


def process_lasso_group(
    osw_path: str,
    ds_ww3: xr.Dataset,
    group_name: str,
    config: dict[str, Any],
    sar_start: datetime,
) -> xr.Dataset | None:
    """Lasso mode: extract WW3 points within buffered footprint of the SAR subswath."""
    logger.info("--- Lasso Mode: %s ---", group_name)

    # 1. Load SAR data
    fat_osw, _ = read_osw(group_name, [Path(osw_path)])
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
        # Shape (subswath, corners, tiles) or (corners, tiles, subswath)
        if lon_corners.shape[0] == 1 and lon_corners.shape[1] == 4:
            # Case: (1, 4, n_tiles) -> (4, n_tiles)
            lon_corners = lon_corners[0]  # (4, n_tiles)
            lat_corners = lat_corners[0]  # (4, n_tiles)

        if lon_corners.ndim == 2 and lon_corners.shape[0] == 4:
            # Now (4, n_tiles) -> transpose to (n_tiles, 4)
            lon_corners = lon_corners.T  # (n_tiles, 4)
            lat_corners = lat_corners.T  # (n_tiles, 4)

    elif lon_corners.ndim == 4:
        # Shape (subswath, az, ra, corners)
        lon_corners = lon_corners.reshape(-1, 4)  # (n_tiles, 4)
        lat_corners = lat_corners.reshape(-1, 4)

    elif lon_corners.ndim == 2 and lon_corners.shape[1] == 4:
        # Already (n_tiles, 4) - perfect
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

    # Compute convex hull and buffer

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

    # Filter WW3 dataset
    ds_ww3_filtered = ds_ww3_subset.isel(time=spatial_mask)

    # Add metadata
    ds_ww3_filtered.attrs["sar_file"] = str(Path(osw_path).name)
    ds_ww3_filtered.attrs["buffer_deg"] = buffer_deg

    logger.info("Selected %d WW3 points out of %d", np.sum(spatial_mask), len(ww3_lons))
    return ds_ww3_filtered


def main() -> None:
    """Main function to process SAR and WW3 data for colocalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--osw-file", required=True, help="path of nc file containing S1 OSW data"
    )
    parser.add_argument(
        "--ww3-file",
        default=None,
        help="path of nc file containing WW3 spectra. If not provided, the script will search based on SAR time and config directory",
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
        "--output-dir", required=True, help="Directory to save output files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files. [optional, default: False]",
        default=False,
    )
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    osw_file_path = Path(args.osw_file)
    fname = osw_file_path.name
    sar_start = datetime.strptime(fname.split("-")[4], "%Y%m%dt%H%M%S").replace(
        tzinfo=timezone.utc
    )
    ww3_path = args.ww3_file or find_ww3_file(sar_start, config)
    output_name = fname.replace(".nc", f"_L2C_{args.mode}.nc")

    ds_ww3 = xr.open_dataset(ww3_path)
    groups = ["intraburst", "interburst"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(output_name)
    logger.info("Output will be saved to: %s", output_path)
    if output_path.exists():
        output_path.unlink()

    if output_path.exists() and not args.overwrite:
        logger.info(
            "Output file %s already exists. Use --overwrite to allow overwriting.",
            output_path,
        )
        return

    if args.mode == "lasso":
        # For lasso mode, process WW3 only for the SAR first group (intraburst if both)
        processed = False
        for g in groups:
            logger.info("Processing lasso for intraburst group")
            fat_osw, _ = read_osw(g, [Path(args.osw_file)])
            if fat_osw is not None:
                if "tiles" in fat_osw.dims or "tiles" in fat_osw.coords:
                    fat_osw_to_write = fat_osw.reset_index("tiles")
                else:
                    fat_osw_to_write = fat_osw

                if not processed:
                    # First write: create file with SAR group
                    fat_osw_to_write.to_netcdf(output_path, group=f"SAR_{g}", mode="w")
                    processed = True
                else:
                    # Subsequent groups: add only SAR (if needed) but skip WW3 duplication
                    fat_osw_to_write.to_netcdf(output_path, group=f"SAR_{g}", mode="a")
                    logger.info("Added SAR_%s without duplicating WW3 data", g)
        ds_ww3_selected = process_lasso_group(
            args.osw_file,
            ds_ww3,
            group_name="intraburst",
            config=config,
            sar_start=sar_start,
        )
        if ds_ww3_selected is not None:
            ds_ww3_selected.to_netcdf(output_path, group="WW3", mode="a")
            logger.info(" WW3 group added to the netcdf")
        if not processed:
            logger.info("No WW3 data extracted")
        return
    first_write = True
    for g in groups:
        result = process_group(args.osw_file, ds_ww3, g, config, sar_start, args.mode)
        d_sar, d_ww3, d_match = result
        # Explicitly check each component to satisfy mypy
        if d_sar is not None and d_ww3 is not None and d_match is not None:
            mode_flag = "w" if first_write else "a"
            logger.info("Writing %s to %s", g, output_path)
            d_sar.to_netcdf(output_path, group=f"SAR_{g}", mode=mode_flag)
            d_ww3_copy = d_ww3.copy()
            d_ww3_copy.encoding.clear()
            d_ww3_copy.to_netcdf(output_path, group=f"WW3_{g}", mode="a")
            d_match.to_netcdf(output_path, group=f"MATCH_MAP_{g}", mode="a")
            first_write = False

    logger.info("Done.")


if __name__ == "__main__":
    main()
