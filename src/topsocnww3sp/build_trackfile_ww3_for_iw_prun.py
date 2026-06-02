#!/usr/bin/env python3
"""Launch prun for trackfile generation."""

import argparse
import datetime
import logging
import subprocess
from pathlib import Path

import numpy as np
from dateutil import rrule

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Configure logging with a named logger
    FMT = "%(asctime)s %(levelname)-5s %(message)s"
    DATEFMT = "%d/%m/%Y %H:%M:%S"
    LEVEL = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=LEVEL, format=FMT, datefmt=DATEFMT)

    PRUNEXE = "/appli/prun/bin/prun"

    listing = Path(
        "/home1/scratch/agrouaze/listing-trackfileww3-IW-filled-on-thefly.txt"
    )
    # Add timezone to make datetimes aware (UTC)
    start = datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
    stop = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)

    # Modify initial listing with 2 more args
    with listing.open("w", encoding="utf-8") as fid:
        content = listing.read_text(encoding="utf-8").splitlines()
        taille = len(content)
        for dd in rrule.rrule(rrule.DAILY, dtstart=start, until=stop):
            ll2 = dd.strftime("%Y%m%d") + "\n"
            fid.write(ll2)

    PBS = "/home1/datahome/agrouaze/sources/projet_sarwave/ww3spectra-trackfile-iw/build-trackfile-ww3-for-iw.pbs"
    # Call prun
    # Use f-string instead of % formatting
    OPTS = f" --split-max-lines={int(np.ceil(taille / 9900.0))} --background -e "

    CMD = PRUNEXE + OPTS + PBS + " " + str(listing)
    logger.info("cmd to cast = %s", CMD)
    STATUS_OUT = subprocess.check_call(CMD, shell=True)
    logger.info("status cmd = %s", STATUS_OUT)
