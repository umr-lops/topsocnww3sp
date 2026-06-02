#!/usr/bin/env python3
"""
Purpose: version of WW3 trackfile generator adapted to S1 ESA TOPS OCN Level-2 osw cross spectra nc files.
"""

import argparse
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import MultiPoint, Point
from tqdm import tqdm

logger = logging.getLogger(__name__)

# WW3 grid definition
lons_ww3 = np.arange(-180, 180, 0.5)
lats_ww3 = np.arange(-80, 80, 0.5)
XX3, YY3 = np.meshgrid(lons_ww3, lats_ww3)
geogridpts = np.stack([XX3.flatten(), YY3.flatten()]).T
points = MultiPoint(geogridpts)


def get_polygon_subswath(dsosw: xr.Dataset) -> shapely.geometry.polygon.Polygon:
    """
    Get polygon of subswath from OSW intraburst group.

    Args:
        dsosw: xarray dataset of OSW intraburst group.

    Returns:
        Shapely polygon of the subswath.
    """
    lons = dsosw["oswLon"].squeeze().to_numpy().ravel()
    lats = dsosw["oswLat"].squeeze().to_numpy().ravel()
    points_list = list(zip(lons, lats, strict=False))
    multi_point = MultiPoint(points_list)
    return multi_point.convex_hull


def collect_each_matching_locations(safedir: str, dirout: str) -> None:
    """
    Collect matching locations between OSW footprints and WW3 grid points.

    Args:
        safedir: Path to S1 OCN SAFE directory containing measurement .nc files.
        dirout: Output directory for .txt files.
    """
    safe_path = Path(safedir)
    pattern = safe_path / "measurement" / "*osw*.nc"
    lst_osw = list(pattern.glob("*"))
    if not lst_osw:
        logger.warning("No OSW files found in %s", safedir)
        return

    base_safe = safe_path.name
    logger.info("SAFE to process: %s", base_safe)
    lst_osw = sorted(lst_osw)
    logger.info("nb L2 OCN osw files to read : %d", len(lst_osw))
    disable_tqdm = len(lst_osw) < 10
    pbar = tqdm(range(len(lst_osw)), disable=disable_tqdm)

    dates = []
    lons_match = []
    lats_match = []
    cpt_unreadable = 0
    unreadable_l1c = []

    for ii in pbar:
        osw_file = lst_osw[ii]
        try:
            dstmp = xr.open_dataset(osw_file, group="intraburst", engine="h5netcdf")
            date_str = osw_file.name.split("-")[5]
            date = datetime.datetime.strptime(date_str, "%Y%m%dt%H%M%S").replace(
                tzinfo=datetime.timezone.utc
            )

            polygon_raw = get_polygon_subswath(dsosw=dstmp)
            fp_dilated = polygon_raw.buffer(0.25)
            valid_points = points.intersection(fp_dilated)

            # valid_points might be a GeometryCollection or MultiPoint
            if hasattr(valid_points, "geoms"):
                pts = valid_points.geoms
            else:
                pts = [valid_points] if not valid_points.is_empty else []
            for individual_pt in pts:
                if isinstance(individual_pt, Point):
                    lons_match.append(individual_pt.x)
                    lats_match.append(individual_pt.y)
                    dates.append(date)
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt without modification
            logger.info("KeyboardInterrupt received, exiting gracefully.")
            raise
        except (OSError, ValueError, KeyError, IndexError, RuntimeError):
            logger.exception("Error processing %s ", osw_file)
            cpt_unreadable += 1
            unreadable_l1c.append(str(osw_file))

    logger.info("unreadable : %d", cpt_unreadable)
    if lons_match:
        logger.info(
            "lon/lat range: %1.2f/%1.2f to %1.2f/%1.2f",
            np.min(lons_match),
            np.min(lats_match),
            np.max(lons_match),
            np.max(lats_match),
        )
        df = pd.DataFrame(
            {
                "lon": lons_match,
                "lat": lats_match,
                "date": dates,
            }
        )
        df["lon_str"] = df["lon"].map(lambda x: f"{x:.2f}")
        df["lat_str"] = df["lat"].map(lambda x: f"{x:.2f}")
        df["YYYYMMDD"] = df["date"].dt.strftime("%Y%m%d")
        df["HHMMSS"] = df["date"].dt.strftime("%H%M%S")
        df = df[["YYYYMMDD", "HHMMSS", "lon_str", "lat_str"]]

        out_dir = Path(dirout)
        out_dir.mkdir(parents=True, exist_ok=True)
        fout = out_dir / f"trackfile-ww3spectra-{base_safe.replace('.SAFE', '')}.txt"
        df.to_csv(fout, header=False, index=False, sep=" ")
        logger.info("fout : %s", fout)
    else:
        logger.info("no data")


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the script."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
    )


def entry_point_one_safe() -> None:
    """Process a single SAFE directory."""
    # Random sleep to avoid mkdir conflicts (use jitter)
    rng = np.random.default_rng()
    time.sleep(rng.random())

    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--OCNSAFE", required=True, help="directory SAFE where to find OCN files"
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    collect_each_matching_locations(safedir=args.OCNSAFE, dirout=args.outputdir)
    logger.info("Processing complete.")


def entry_point_one_listing_of_safe() -> None:
    """Process a list of SAFE directories from a text file."""
    rng = np.random.default_rng()
    time.sleep(rng.random())

    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--listing-safe", required=True, help="path of a listing of OCN SAFE files"
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    listing_path = Path(args.listing_safe)
    safes = pd.read_csv(listing_path, header=None)[0].tolist()
    for ss in tqdm(safes):
        logger.info("processing SAFE: %s", ss)
        collect_each_matching_locations(safedir=ss, dirout=args.outputdir)
    logger.info("Output files generated in %s", args.outputdir)


def entry_point_ocn_between_dates() -> None:
    """Process all SAFE directories between two dates."""
    rng = np.random.default_rng()
    time.sleep(rng.random())

    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--inputdir",
        required=True,
        help="directory where OCN are stored (up to .../S1A_IW_OCN__2S/)",
    )
    parser.add_argument("--start", required=True, help="start date YYYYMMDD inclusive")
    parser.add_argument("--end", required=True, help="end date YYYYMMDD inclusive")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    input_dir = Path(args.inputdir)
    safes = []
    for day in pd.date_range(start=args.start, end=args.end):
        pattern = (
            input_dir
            / day.strftime("%Y")
            / day.strftime("%m")
            / day.strftime("%d")
            / "S1*_OCN__2S*.SAFE"
        )
        safes.extend(list(pattern.glob("*")))
    logger.info("number of SAFE to process: %d", len(safes))
    for ss in tqdm(safes):
        logger.info("processing SAFE: %s", ss)
        collect_each_matching_locations(safedir=str(ss), dirout=args.outputdir)
    logger.info("Processing complete.")


if __name__ == "__main__":
    # entry_point_one_safe()
    entry_point_one_listing_of_safe()
    # entry_point_ocn_between_dates()
