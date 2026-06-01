#!/usr/bin/env python3
"""Count WW3 spectra associated with each OCN tile from a trackfile."""

import argparse
import logging
import typing
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from topsocnww3sp.utils import get_config

logger = logging.getLogger(__name__)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> np.ndarray:
    """Calculate great circle distance in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371


def parse_track_file(filepath: str) -> list[dict[str, typing.Any]]:
    """Parses the text track file into a list of dictionaries."""
    data = []
    logger.info("Reading track file: %s", filepath)
    path_obj = Path(filepath)
    try:
        with path_obj.open() as f:
            # Skip header line (unused)
            _ = f.readline()
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 4:
                    continue
                dt_str = f"{parts[0]} {parts[1]}"
                try:
                    # Directly create aware datetime
                    dt = datetime.strptime(dt_str, "%Y%m%d %H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                    data.append(
                        {
                            "line_idx": i + 2,
                            "datetime": dt,
                            "longitude": float(parts[2]),
                            "latitude": float(parts[3]),
                        }
                    )
                except ValueError:
                    continue
    except FileNotFoundError:
        logger.exception("Trackfile not found: %s", filepath)
    return data


def core_count_coverage(
    track_points: list[dict[str, typing.Any]],
    ww3_nc_lons: np.ndarray,
    ww3_nc_lats: np.ndarray,
    ww3_nc_times: np.ndarray,
    pathconfig: str | None = None,
) -> tuple[list[str], dict[int, int]]:
    """
    Arguments:
        track_points: List of dicts with keys 'line_idx', 'datetime', 'longitude', 'latitude'
        ww3_nc_lons: 1D array of longitudes from WW3 NetCDF
        ww3_nc_lats: 1D array of latitudes from WW3 NetCDF
        ww3_nc_times: 1D array of datetimes from WW3 NetCDF
        pathconfig: Optional path to config.yml for thresholds

    Returns:
        summary_lines: List of strings summarizing the distribution
        results: Dict mapping track line index to count of associated spectra
    """
    config = get_config(path_config=pathconfig)
    results = {}
    time_delta = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])

    # 3. Matching Loop
    logger.info("Starting matching process...")
    for pt in tqdm(track_points, desc="Matching"):
        t_start, t_end = pt["datetime"] - time_delta, pt["datetime"] + time_delta

        # Temporal mask
        time_mask = (ww3_nc_times >= t_start) & (ww3_nc_times <= t_end)

        if not np.any(time_mask):
            results[pt["line_idx"]] = 0
            continue

        # Spatial distance for points in time window
        distances = haversine(
            pt["longitude"],
            pt["latitude"],
            ww3_nc_lons[time_mask],
            ww3_nc_lats[time_mask],
        )

        results[pt["line_idx"]] = int(
            np.sum(distances <= config["DISTANCE_THRESHOLD_KM"])
        )

    # 4. Distribution and Statistics Calculation
    dist_counts = Counter(results.values())
    total_tiles = len(track_points)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(
        f"{'DISTRIBUTION OF WW3 SPECTRA ASSOCIATED PER OCN TILE within {} minutes and {} km':^80}".format(
            config["TIME_THRESHOLD_MINUTES"], config["DISTANCE_THRESHOLD_KM"]
        )
    )
    summary_lines.append("=" * 80)
    summary_lines.append(
        f"{'Spectra Count':<20} | {'Trackfile Points':<15} | {'Percentage':<15}"
    )
    summary_lines.append("-" * 80)

    # Sort by number of spectra (0, 1, 2...)
    for count in sorted(dist_counts.keys()):
        freq = dist_counts[count]
        pct = (freq / total_tiles) * 100
        summary_lines.append(f"{count:<20} | {freq:<15} | {pct:>6.2f}%")

    summary_lines.append("-" * 80)
    summary_lines.append(f"{'TOTAL':<20} | {total_tiles:<15} | 100.00%")
    summary_lines.append("=" * 80)

    return summary_lines, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match WW3 track output with a trackfile list."
    )
    parser.add_argument("-t", "--trackfile", required=True)
    parser.add_argument("-n", "--ncfiles", nargs="+", required=True)
    parser.add_argument("-o", "--output", default="counts_report.txt")
    parser.add_argument("--config", default=None, help="Path to config.yml (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # No global logger needed; we just use the module-level logger
    # The logger was already defined at module level

    # 1. Load Trackfile
    track_points = parse_track_file(args.trackfile)
    if not track_points:
        return
    # 2. Open NetCDF datasets
    logger.info("Opening %d NetCDF file(s)...", len(args.ncfiles))
    try:
        ds = xr.open_mfdataset(
            args.ncfiles, combine="nested", concat_dim="time", decode_times=True
        )
        ww3_nc_lons = ds.longitude.to_numpy()
        ww3_nc_lats = ds.latitude.to_numpy()
        ww3_nc_times = pd.to_datetime(ds.time.to_numpy())
    except Exception:
        logger.exception("Failed to process NetCDF")
        return

    summary_lines, results = core_count_coverage(
        track_points, ww3_nc_lons, ww3_nc_lats, ww3_nc_times, pathconfig=args.config
    )
    # 5. Write to Output File
    output_path = Path(args.output)
    logger.info("Writing detailed report to %s", output_path)
    with output_path.open("w") as out:
        out.write("\n".join(summary_lines) + "\n\n")
        out.write("DETAILED DATA:\n")
        out.write(
            "Line_Number | DateTime | Longitude | Latitude | Spectra_Match_Count\n"
        )
        out.write("-" * 80 + "\n")
        for pt in track_points:
            idx = pt["line_idx"]
            out.write(
                f"{idx:<11} | {pt['datetime']} | {pt['longitude']:>9.3f} | {pt['latitude']:>8.3f} | {results[idx]}\n"
            )


if __name__ == "__main__":
    main()
