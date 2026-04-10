import argparse
import glob
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from topsocnww3sp.read_s1_osw_tops_data import read_osw

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def find_ww3_file(sar_time, config):
    base_dir = config["directory_ww3spectra_output"]
    year_month = sar_time.strftime("%Y%m")
    search_pattern = os.path.join(base_dir, "**", f"*_{year_month}_trck.nc")
    found_files = glob.glob(search_pattern, recursive=True)
    if not found_files:
        raise FileNotFoundError(f"No WW3 file for {year_month} in {base_dir}")
    return found_files[0]


def process_group(osw_path, ds_ww3, group_name, config, sar_start, mode):
    logger.info(f"--- Processing Group: {group_name} [Mode: {mode}] ---")

    # 1. Load and Flatten SAR
    fat_osw, _ = read_osw(group_name, [osw_path])
    if fat_osw is None or len(fat_osw.data_vars) == 0:
        return None, None, None

    fat_osw["time"] = sar_start
    sar_flat = (
        fat_osw.reset_index("tiles")
        .stack(all_tiles=["subswath", "tiles"])
        .dropna("all_tiles", subset=["oswLon"])
    )
    n_tiles = len(sar_flat.all_tiles)

    # 2. Temporal Filter WW3
    ww3_times = pd.to_datetime(ds_ww3.time.values)
    t_sar = pd.to_datetime(sar_start)
    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)

    if not np.any(t_mask):
        return None, None, None

    cand_idx_orig = np.where(t_mask)[0]
    cand_lons = ds_ww3.longitude.values[t_mask]
    cand_lats = ds_ww3.latitude.values[t_mask]
    dist_thresh = config["DISTANCE_THRESHOLD_KM"]

    # 3. Matchup Logic
    sar_indices, ww3_indices_rel, distances = [], [], []

    for i in range(n_tiles):
        tile = sar_flat.isel(all_tiles=i)
        dists = haversine(tile.oswLon.values, tile.oswLat.values, cand_lons, cand_lats)

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
        actual_ww3_times = ds_ww3_sel.time.values
        # 3. Rename dimension to align with SAR tiles
        ds_ww3_out = ds_ww3_sel.rename({"time": "all_tiles"})
        # 4. Re-assign simple integer coordinates to all_tiles
        ds_ww3_out = ds_ww3_out.assign_coords(all_tiles=tile_coords)
        # 5. Put the timestamps back into a variable named 'time'
        ds_ww3_out["time"] = (["all_tiles"], actual_ww3_times)

        # Calculate time diff
        # We handle NaTs (for non-matches) gracefully
        ww3_dt = pd.to_datetime(actual_ww3_times)
        t_diffs = (ww3_dt - t_sar).total_seconds().values

        ds_match = xr.Dataset(
            {
                "distance_km": (
                    ["all_tiles"],
                    pd.Series(distances, index=sar_indices).reindex(tile_coords).values,
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
                    pd.Series(distances, index=sar_indices).reindex(tile_coords).values,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osw-file", required=True)
    parser.add_argument("--ww3-file", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["1to1", "unique", "many"], default="1to1")
    parser.add_argument(
        "--group", choices=["intraburst", "interburst", "both"], default="both"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fname = os.path.basename(args.osw_file)
    sar_start = datetime.strptime(fname.split("-")[4], "%Y%m%dt%H%M%S")
    ww3_path = args.ww3_file or find_ww3_file(sar_start, config)
    output_name = fname.replace(".nc", f"_L2C_{args.mode}.nc")

    ds_ww3 = xr.open_dataset(ww3_path)
    groups = ["intraburst", "interburst"] if args.group == "both" else [args.group]

    if os.path.exists(output_name):
        os.remove(output_name)
    first_write = True

    for g in groups:
        res = process_group(args.osw_file, ds_ww3, g, config, sar_start, args.mode)
        if res[0] is not None:
            d_sar, d_ww3, d_match = res
            mode_flag = "w" if first_write else "a"
            logger.info(f"Writing {g} to {output_name}")
            d_sar.to_netcdf(output_name, group=f"SAR_{g}", mode=mode_flag)
            d_ww3.to_netcdf(output_name, group=f"WW3_{g}", mode="a")
            d_match.to_netcdf(output_name, group=f"MATCH_MAP_{g}", mode="a")
            first_write = False

    logger.info("Done.")


if __name__ == "__main__":
    main()
