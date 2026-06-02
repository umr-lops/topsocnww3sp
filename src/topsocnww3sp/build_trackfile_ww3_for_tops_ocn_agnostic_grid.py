#!/usr/bin/env python3
"""
Purpose: version of WW3 trackfile generator adapted to S1 ESA TOPS OCN Level-2 osw cross spectra nc files.
Aggregates all positions from multiple SAFEs into a single globally sorted trackfile.
"""

import argparse
import datetime
import logging
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from topsocnww3sp.utils import format_logs

logger = logging.getLogger(__name__)


def _parse_single_osw(osw_file: Path) -> tuple[list[dict[str, Any]], bool]:
    """
    Helper to parse a single OSW NC file.
    """
    flag_file_parsed = True
    positions = []
    try:
        if "ew" in str(osw_file) or "iw" in str(osw_file):
            with xr.open_dataset(
                osw_file, group="intraburst", engine="h5netcdf"
            ) as dstmp:
                # Extract timestamp from filename
                date_part = osw_file.name.split("-")[5]
                date = datetime.datetime.strptime(date_part, "%Y%m%dt%H%M%S").replace(
                    tzinfo=datetime.timezone.utc
                )

                lons = dstmp["oswLon"].squeeze().to_numpy().ravel()
                lats = dstmp["oswLat"].squeeze().to_numpy().ravel()

                for lon, lat in zip(lons, lats, strict=False):
                    positions.append({"lon": lon, "lat": lat, "date": date})
        else:  # wv 'case'
            assert "wv" in str(osw_file)
            with xr.open_dataset(osw_file, engine="h5netcdf") as dstmp:
                # Extract timestamp from filename
                date_part = osw_file.name.split("-")[5]
                date = datetime.datetime.strptime(date_part, "%Y%m%dt%H%M%S").replace(
                    tzinfo=datetime.timezone.utc
                )

                lons = dstmp["oswLon"].squeeze().to_numpy().ravel()
                lats = dstmp["oswLat"].squeeze().to_numpy().ravel()

                for lon, lat in zip(lons, lats, strict=False):
                    positions.append({"lon": lon, "lat": lat, "date": date})
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        logger.exception("Error processing %s: %s", osw_file, traceback.format_exc())
        flag_file_parsed = False
    return positions, flag_file_parsed


def collect_positions_from_safe(
    safedir: str, counter: defaultdict
) -> tuple[list[dict[str, Any]], defaultdict]:
    """
    Collect all positions from OCN files within a SAFE directory.
    """
    safe_path = Path(safedir)
    if "EW" in safedir or "IW" in safedir:
        pattern_osw = "measurement/*osw*.nc"
    else:
        pattern_osw = "measurement/*wv*.nc"
    # Remove unnecessary list() call
    lst_osw = sorted(safe_path.glob(pattern_osw))

    logger.debug("SAFE: %s (%d files)", safe_path.name, len(lst_osw))

    safe_positions: list[dict[str, Any]] = []
    for osw_file in lst_osw:
        positions, flag_parsed = _parse_single_osw(osw_file)
        if flag_parsed:
            safe_positions.extend(positions)
            counter["nc_file_read"] += 1
        else:
            counter["nc_file_erro"] += 1

    return safe_positions, counter


def write_aggregated_trackfile(
    all_positions: list[dict[str, Any]], dirout: str | Path, dev: bool = False
) -> None:
    """
    Sorts positions, generates filename based on date range, and writes to disk.
    """
    if not all_positions:
        logger.warning("No positions collected. Nothing to write.")
        return

    df = pd.DataFrame(all_positions)
    # Global Sort
    df = df.sort_values(by="date")

    first_date = df["date"].min().strftime("%Y%m%dt%H%M%S")
    last_date = df["date"].max().strftime("%Y%m%dt%H%M%S")

    # Format trackfile columns using f-strings
    df["YYYYMMDD"] = df["date"].dt.strftime("%Y%m%d")
    df["HHMMSS"] = df["date"].dt.strftime("%H%M%S")
    df["lon_str"] = df["lon"].map(lambda x: f"{x:.2f}")
    df["lat_str"] = df["lat"].map(lambda x: f"{x:.2f}")

    output_df = df[["YYYYMMDD", "HHMMSS", "lon_str", "lat_str"]]

    out_path = Path(dirout)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = "trackfile-ww3spectra-agnostic"
    if dev:
        prefix += "-DEV"

    fout = out_path / f"{prefix}-{first_date}-{last_date}.txt"

    output_df.to_csv(fout, header=False, index=False, sep=" ")
    logger.info("Agnostic trackfile saved: %s", fout)


def entry_point_one_listing_of_safe() -> None:
    """Process multiple SAFE directories from a listing file into ONE output."""
    rng = np.random.default_rng()
    time.sleep(rng.uniform(0, 1))

    parser = argparse.ArgumentParser(
        description="Agnostic aggregated trackfile generator"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Development mode: process only first 3 SAFEs",
    )
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--listing-safe", required=True, help="path of a listing of OCN SAFE files"
    )
    args = parser.parse_args()

    format_logs(logging.getLogger(__name__), "debug" if args.verbose else "info")

    t0 = time.time()

    # Load listing using Path.open()
    listing_path = Path(args.listing_safe)
    with listing_path.open(encoding="utf-8") as f:
        safes = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    # DEV Mode handling
    if args.dev:
        logger.info("--- DEVELOPMENT MODE ACTIVE: Limiting input to first 3 SAFEs ---")
        safes = safes[:3]

    all_aggregated_positions: list[dict[str, Any]] = []

    # Aggregation Loop
    pbar = tqdm(safes, desc="Aggregating SAFEs")
    counter: defaultdict[str, int] = defaultdict(int)
    for safe_dir in pbar:
        # Use Path(...).name instead of os.path.basename
        mode = Path(safe_dir.rstrip("/")).name.split("_")[1]
        counter[mode] += 1
        pbar.set_description(f"trackfile generation / {counter}")
        addition_positions, counter = collect_positions_from_safe(safe_dir, counter)
        counter["total_positions"] += len(addition_positions)
        all_aggregated_positions.extend(addition_positions)

    # Single Write
    write_aggregated_trackfile(all_aggregated_positions, args.outputdir, dev=args.dev)
    logger.info("counter final: %s", counter)
    elapsed = time.time() - t0
    logger.info("Total processing time: %1.1f seconds", elapsed)


if __name__ == "__main__":
    entry_point_one_listing_of_safe()
