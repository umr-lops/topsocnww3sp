#!/usr/bin/env python3
"""Concatenate trackfiles from SAFE OCN processing into a single file."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _read_single_trackfile(trackfile: Path) -> pd.DataFrame | None:
    """
    Read a single trackfile and return a DataFrame, or None if error.
    """
    try:
        # Return directly to avoid RET504 and TRY300
        return pd.read_csv(
            trackfile,
            sep=r"\s+",
            header=None,
            names=["YYYYMMDD", "HHMMSS", "lon_str", "lat_str"],
            dtype={"YYYYMMDD": int, "HHMMSS": int},
        )
    except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
        logger.exception("Error reading %s", trackfile)
        return None


def concatenate_trackfiles(input_dir: str, output_file: str) -> None:
    """Concatenate all trackfile-ww3spectra-agnostic-*.txt files into one.

    Args:
        input_dir: Directory containing the trackfiles.
        output_file: Path to the output concatenated file.
    """
    input_path = Path(input_dir)
    pattern = "trackfile-ww3spectra-agnostic-*.txt"
    trackfiles = list(input_path.glob(pattern))

    if not trackfiles:
        logger.warning("No trackfiles found in %s", input_dir)
        return

    all_data = []

    for trackfile in tqdm(trackfiles, desc="Reading trackfiles"):
        df = _read_single_trackfile(trackfile)
        if df is not None:
            all_data.append(df)

    if not all_data:
        logger.warning("No valid data frames to concatenate.")
        return

    # 1. Concatenate
    combined_df = pd.concat(all_data, ignore_index=True)

    # 2. Sort by Date AND Time columns globally
    logger.info("Sorting data chronologically...")
    combined_df = combined_df.sort_values(by=["YYYYMMDD", "HHMMSS"]).reset_index(
        drop=True
    )

    # 3. Save to output
    output_path = Path(output_file)
    logger.info("Saving to %s", output_path)
    with output_path.open("w") as f:
        f.write("WAVEWATCH III TRACK LOCATIONS DATA \n")
        # Write each row with proper formatting
        for _, row in combined_df.iterrows():
            line = (
                f"{int(row['YYYYMMDD']):8d} "
                f"{int(row['HHMMSS']):06d} "
                f"{float(row['lon_str']):10.5f} "
                f"{float(row['lat_str']):10.5f}\n"
            )
            f.write(line)

    logger.info("Done. Total records: %d", len(combined_df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate trackfile-ww3spectra-agnostic-*.txt files."
    )
    parser.add_argument("--input-dir", required=True, help="Directory with trackfiles")
    parser.add_argument("--output-file", required=True, help="Output concatenated file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    concatenate_trackfiles(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
