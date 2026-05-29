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

from topsocnww3sp.read_s1_osw_tops_data import read_osw

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    Args:
        lon1: Longitude of first point in decimal degrees
        lat1: Latitude of first point in decimal degrees
        lon2: Longitude of second point in decimal degrees
        lat2: Latitude of second point in decimal degrees

    Returns:
        Distance between the two points in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


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
    fat_osw, _ = read_osw(group_name, [osw_path])
    if fat_osw is None or len(fat_osw.data_vars) == 0:
        return None, None, None

    # fat_osw["time"] = sar_start
    fat_osw["time"] = np.datetime64(sar_start.replace(tzinfo=None), 'ns')

# First reset the existing MultiIndex on tiles, then stack all spatial dims flat
    ds_reset = fat_osw.reset_index("tiles")
    ds_stacked = ds_reset.stack(all_tiles=("subswath", "tiles"))
    valid_mask = ~np.isnan(ds_stacked["oswLon"].values)
    sar_flat = ds_stacked.isel(all_tiles=valid_mask)
    sar_flat = sar_flat.drop_vars(['all_tiles', 'subswath', 'tiles']).assign_coords(
        all_tiles=np.arange(len(sar_flat.all_tiles))
    )

    n_tiles = len(sar_flat.all_tiles)

    # 2. Temporal Filter WW3
    ww3_times = pd.to_datetime(ds_ww3.time.to_numpy())
    t_sar = pd.Timestamp(sar_start).tz_localize(None)  # strip tzinfo for comparison
    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)

    if not np.any(t_mask):
        return None, None, None

    cand_idx_orig = np.where(t_mask)[0]
    cand_lons = ds_ww3.longitude.to_numpy()[t_mask]
    cand_lats = ds_ww3.latitude.to_numpy()[t_mask]
    dist_thresh = config["DISTANCE_THRESHOLD_KM"]

    # 3. Matchup Logic
    sar_indices: list[int] = []
    ww3_indices_rel: list[int] = []
    distances: list[float] = []

    for i in range(n_tiles):
        tile = sar_flat.isel(all_tiles=i)
        dists = haversine(
            tile.oswLon.to_numpy(), tile.oswLat.to_numpy(), cand_lons, cand_lats
        )

        if mode in ["1to1", "unique"]:
            min_idx = np.argmin(dists)
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

    # Ensure SAR group also has clean integer indices
    ds_sar = sar_flat.reset_index("all_tiles").assign_coords(all_tiles=tile_coords)

    return ds_sar, ds_ww3_out, ds_match


def main() -> None:
    """Main function to process SAR and WW3 data for colocalization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--osw-file", required=True,
                        help="path of nc file containing S1 OSW data")
    parser.add_argument("--ww3-file", default=None,
                        help="path of nc file containing WW3 spectra. If not provided, the script will search based on SAR time and config directory")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file with thresholds and directories")
    parser.add_argument("--mode", choices=["1to1", "unique", "many"], default="1to1",
                        help="Matching mode: '1to1' (one WW3 per SAR), 'unique' (multiple SAR can share same WW3), or 'many' (all matches)"
    )
    parser.add_argument(
        "--group", choices=["intraburst", "interburst", "both"], default="both",
        help="Which group(s) to process: 'intraburst', 'interburst', or 'both' (default: both)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging"
        )
    parser.add_argument('--output-dir',required=True, help='Directory to save output files.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Whether to overwrite existing output files. [optional, default: False]',default=False)
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)


    config_path = Path(args.config)
    config = yaml.safe_load(config_path.open())

    osw_file_path = Path(args.osw_file)
    fname = osw_file_path.name
    sar_start = datetime.strptime(fname.split("-")[4], "%Y%m%dt%H%M%S").replace(
        tzinfo=timezone.utc
    )
    ww3_path = args.ww3_file or find_ww3_file(sar_start, config)
    output_name = fname.replace(".nc", f"_L2C_{args.mode}.nc")

    ds_ww3 = xr.open_dataset(ww3_path)
    groups = ["intraburst", "interburst"] if args.group == "both" else [args.group]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path= output_dir / Path(output_name)
    logger.info("Output will be saved to: %s", output_path)
    if output_path.exists():
        output_path.unlink()

    if output_path.exists() and not args.overwrite:
        logger.info("Output file %s already exists. Use --overwrite to allow overwriting.", output_path)
        return
    
    first_write = True

    for g in groups:
        res = process_group(args.osw_file, ds_ww3, g, config, sar_start, args.mode)
        if res[0] is not None:
            d_sar, d_ww3, d_match = res
            mode_flag = "w" if first_write else "a"
            logger.info("Writing %s to %s", g, output_path)
            d_sar.to_netcdf(output_path, group=f"SAR_{g}", mode=mode_flag)
            d_ww3 = d_ww3.copy()
            d_ww3.encoding.clear()
            d_ww3.to_netcdf(output_path, group=f"WW3_{g}", mode="a")
            d_match.to_netcdf(output_path, group=f"MATCH_MAP_{g}", mode="a")
            first_write = False

    logger.info("Done.")


if __name__ == "__main__":
    main()
