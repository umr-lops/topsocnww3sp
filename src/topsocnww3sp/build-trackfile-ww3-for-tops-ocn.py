#!/usr/bin/python
# encoding *-utf-8-*
"""
purpose: version of WW3 trackfile generator adapted to S1 ESA TOPS OCN Level-2 osw cross spectra nc files
"""

import argparse
import datetime
import glob
import logging
import os
import time
import traceback

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import MultiPoint
from tqdm import tqdm

lons_ww3 = np.arange(
    -180, 180, 0.5
)  # based on /home/ref-ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/2022/FIELD_NC
# lats_ww3 = np.arange(-78,83.5,0.5)
lats_ww3 = np.arange(-80, 80, 0.5)  # lat max to be confirmed by Mickael Accensi
print(lons_ww3.shape, lats_ww3.shape)
XX3, YY3 = np.meshgrid(lons_ww3, lats_ww3)
geogridpts = np.stack([XX3.flatten(), YY3.flatten()]).T
points = MultiPoint(geogridpts)


def get_polygon_subswath(dsosw: xr.Dataset) -> shapely.geometry.polygon.Polygon:
    """

    method to get polygon of subswath from osw intraburst group

    :param dsosw: xarray dataset of osw intraburst group
    :return: shapely polygon of the subswath
    """
    # use convex hull method from shapely to create polygon with each grid points
    lons = dsosw["oswLon"].squeeze().values.ravel()
    lats = dsosw["oswLat"].squeeze().values.ravel()
    # Créer une liste de points (lon, lat)
    points = list(zip(lons, lats))

    # Créer un MultiPoint et calculer l'enveloppe convexe
    multi_point = MultiPoint(points)
    convex_hull_polygon = multi_point.convex_hull
    return convex_hull_polygon


def collect_each_matching_locations(safedir: str, dirout: str) -> None:
    """


    :param safedir : str path of S1 OCN SAFE where to find measurement (.nc) files
    :param dirout: str path where to store the .txt files

    :return:
    """
    # patl1b = os.path.join(dir_l1b,'*SAFE','*vv*.nc')
    # pattern_osw = os.path.join(safedir,day_to_treat.strftime('%Y'),day_to_treat.strftime('%j'),unitsar+'*.SAFE','*osw*1sdv*.nc')
    pattern_osw = os.path.join(safedir, "measurement", "*osw*.nc")
    lst_osw = glob.glob(pattern_osw)
    base_safe = os.path.basename(safedir)
    logging.info("SAFE to process: %s", base_safe)
    lst_osw = sorted(lst_osw)
    logging.info("nb L2 OCN osw files to read : %s", len(lst_osw))
    pbar = tqdm(range(len(lst_osw)), disable=True if len(lst_osw) < 10 else False)
    df = pd.DataFrame()
    dates = []
    lons_match = []
    lats_match = []
    cpt_undreadable = 0
    # unreadable_l1b = []
    unreadable_l1c = []
    for ii in pbar:
        try:
            dstmp = xr.open_dataset(lst_osw[ii], group="intraburst", engine="h5netcdf")
            date = datetime.datetime.strptime(
                os.path.basename(lst_osw[ii]).split("-")[5], "%Y%m%dt%H%M%S"
            )
            polygon_raw = get_polygon_subswath(dsosw=dstmp)
            fp_dilated = polygon_raw.buffer(0.25)
            valid_points = points.intersection(fp_dilated)
            for individualpt in valid_points.geoms:
                lons_match.append(individualpt.x)
                lats_match.append(individualpt.y)
                dates.append(date)
        except KeyboardInterrupt:
            raise Exception("stoop")
        except:
            logging.exception("%s", traceback.format_exc())
            cpt_undreadable += 1
            unreadable_l1c.append(lst_osw[ii])
        if ii == 4:
            pass
    logging.info("unreadable : %s", cpt_undreadable)
    if len(lons_match) > 0:
        logging.info(
            "lon/lat range: %1.2f/%1.2f to %1.2f/%1.2f",
            np.min(lons_match),
            np.min(lats_match),
            np.max(lons_match),
            np.max(lats_match),
        )
        df["lon"] = lons_match
        df["lat"] = lats_match
        # create new colmuns lon_str and lat_str to have consistent number of decimals %3.2f
        df["lon_str"] = df["lon"].map(lambda x: "%.2f" % x)
        df["lat_str"] = df["lat"].map(lambda x: "%.2f" % x)
        df["date"] = dates
        # split date into YYYYMMDD and HHMMSS columns
        df["YYYYMMDD"] = df["date"].dt.strftime("%Y%m%d")
        df["HHMMSS"] = df["date"].dt.strftime("%H%M%S")
        # reorder columns
        df = df[["YYYYMMDD", "HHMMSS", "lon_str", "lat_str"]]
        os.makedirs(dirout, exist_ok=True)
    if len(dates) > 0:
        fout = os.path.join(
            dirout, "trackfile-ww3spectra-%s.txt" % (base_safe.replace(".SAFE", ""))
        )
        df.to_csv(fout, header=False, index=False, sep=" ")
        logging.info("fout : %s", fout)
    else:
        logging.info("no data")


def entry_point_one_safe() -> None:
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    # parser.add_argument('--day', required=True, help='YYYYMMDD')
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--OCNSAFE", required=True, help="directory SAFE where to find OCN files"
    )
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()
    collect_each_matching_locations(safedir=args.OCNSAFE, dirout=args.outputdir)
    elapsed = time.time()
    logging.info("time to do a day: %1.1f seconds", elapsed)


def entry_point_one_listing_of_safe() -> None:
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    # parser.add_argument('--day', required=True, help='YYYYMMDD')
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--listing-safe", required=True, help="path of a listing of OCN SAFE files"
    )
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()
    safes = pd.read_csv(args.listing_safe, header=None)[0].tolist()  # ,names=['safe']
    for ss in tqdm(range(len(safes))):
        logging.info("processing SAFE: %s", safes[ss])

        collect_each_matching_locations(safedir=safes[ss], dirout=args.outputdir)
    elapsed = time.time()
    logging.info("check out the txt files generated in %s", args.outputdir)
    logging.info("time to do a day: %1.1f seconds", elapsed)


def entry_point_ocn_between_dates() -> None:
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description="trackfileiwWW3")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--outputdir", required=True, help="directory where to store output"
    )
    parser.add_argument(
        "--inputdir",
        required=True,
        help="directory where to OCN are stored. up to e.g. .../S1A_IW_OCN__2S/ ",
    )
    parser.add_argument("--start", required=True, help="start date YYYYMMDD inclusive")
    parser.add_argument("--end", required=True, help="end date YYYYMMDD inclusive")
    args = parser.parse_args()
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()
    safes = []
    for day in pd.date_range(start=args.start, end=args.end):
        pattern_safe = os.path.join(
            args.inputdir,
            day.strftime("%Y"),
            day.strftime("%m"),
            day.strftime("%d"),
            "S1*_OCN__2S*.SAFE",
        )
        lst_safe = glob.glob(pattern_safe)
        safes.extend(lst_safe)
    logging.info("number of SAFE to process: %s", len(safes))
    for ss in tqdm(range(len(safes))):
        logging.info("processing SAFE: %s", safes[ss])

        collect_each_matching_locations(safedir=safes[ss], dirout=args.outputdir)
    elapsed = time.time()
    logging.info("time to do a day: %1.1f seconds", elapsed)


if __name__ == "__main__":
    # entry_point_one_safe()
    entry_point_one_listing_of_safe()  # used for IPF401 trackfiles generation : python build-trackfile-ww3-for-tops-ocn.py --listing-safe /home/datawork-cersat-public//cache/project/mpc-sentinel1/data/esl/cls/dataset_validation_xspecTops_ipf401/listing_safe_ocn.txt --outputdir /raid/localscratch/agrouaze/test_trackfile_ocn
    # entry_point_ocn_between_dates()
