#!/usr/bin/env python3
import argparse
import datetime
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import MultiPoint
from shapely.wkt import loads
from tqdm import tqdm

# LOG015: Use a named logger instead of the root logger
logger = logging.getLogger(__name__)

# Grid setup
lons_ww3 = np.arange(-180, 180, 0.5)
lats_ww3 = np.arange(-78, 83.5, 0.5)
XX3, YY3 = np.meshgrid(lons_ww3, lats_ww3)
geogridpts = np.stack([XX3.flatten(), YY3.flatten()]).T
points = MultiPoint(geogridpts)


def _process_single_l1c(
    l1c_file: Path,
) -> tuple[list[float], list[float], list[datetime.datetime]] | None:
    """
    Helper function to process one file.
    """
    lons, lats, dates = [], [], []
    try:
        with xr.open_dataset(l1c_file, group="intraburst", engine="h5netcdf") as dstmp:
            date_str = l1c_file.name.split("-")[5]
            date = datetime.datetime.strptime(date_str, "%Y%m%dt%H%M%S").replace(
                tzinfo=datetime.timezone.utc
            )

            fp_dilated = loads(dstmp.attrs["footprint"]).buffer(0.25)
            valid_points = points.intersection(fp_dilated)

            if not valid_points.is_empty:
                geoms = (
                    valid_points.geoms
                    if hasattr(valid_points, "geoms")
                    else [valid_points]
                )
                for individualpt in geoms:
                    lons.append(individualpt.x)
                    lats.append(individualpt.y)
                    dates.append(date)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        logger.exception("Error reading %s: %s", l1c_file, traceback.format_exc())
        return None

    return lons, lats, dates


def collect_each_matching_locations(
    day_to_treat: datetime.datetime, product_id: str, dirout: str | Path
) -> None:
    """
    Collect matching locations for WW3 trackfile.
    """
    dir_l1c = Path(
        "/home/datawork-cersat-public/project/sarwave/data/products/tests2/slc/iw/l1c/"
    )
    out_path = Path(dirout)

    search_pattern = (
        f"{day_to_treat.strftime('%Y')}/"
        f"{day_to_treat.strftime('%j')}/"
        f"*{product_id}.SAFE/*1sdv*.nc"
    )

    lst_l1c = sorted(dir_l1c.glob(search_pattern))
    logger.info("nb L1C files to read : %s", len(lst_l1c))

    dates: list[datetime.datetime] = []
    lons_match: list[float] = []
    lats_match: list[float] = []
    cpt_unreadable: int = 0
    unreadable_l1c: list[str] = []

    # PERF203: Move try-except KeyboardInterrupt outside the for-loop
    try:
        for l1c_file in tqdm(lst_l1c, desc="Processing L1C files"):
            result = _process_single_l1c(l1c_file)
            if result:
                lons, lats, dts = result
                lons_match.extend(lons)
                lats_match.extend(lats)
                dates.extend(dts)
            else:
                cpt_unreadable += 1
                unreadable_l1c.append(str(l1c_file))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    logger.info("unreadable : %s", cpt_unreadable)

    if dates:
        df = pd.DataFrame({"lon": lons_match, "lat": lats_match, "date": dates})
        out_path.mkdir(parents=True, exist_ok=True)
        fout = (
            out_path
            / f"trackfile-ww3spectra-IW-{day_to_treat.strftime('%Y%m%d')}-{product_id}.txt"
        )
        df["date_str"] = df["date"].dt.strftime("%Y%m%d %H%M%S")
        df[["date_str", "lon", "lat"]].to_csv(fout, header=False, index=False, sep=" ")
        logger.info("fout : %s", fout)
    else:
        logger.info("no data found")


def main() -> None:
    """Main entry point for the script."""
    rng = np.random.default_rng()
    time.sleep(rng.uniform(0, 1))

    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--day", required=True, help="YYYYMMDD")
    parser.add_argument("--outputdir", required=True, help="dir where to store output")
    parser.add_argument("--productid", help="proudct ID", required=False, default="B07")
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
    )

    day_to_treat = datetime.datetime.strptime(args.day, "%Y%m%d").replace(
        tzinfo=datetime.timezone.utc
    )

    collect_each_matching_locations(
        day_to_treat=day_to_treat, product_id=args.productid, dirout=args.outputdir
    )


if __name__ == "__main__":
    main()
