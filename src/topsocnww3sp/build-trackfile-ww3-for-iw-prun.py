#!/scale/project/lops-siam-airflow/envs_exploit/micromamba/py27/bin/python
""" """

import datetime
import logging
import subprocess

import numpy as np
from dateutil import rrule

if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    prunexe = "/appli/prun/bin/prun"

    listing = "/home1/scratch/agrouaze/listing-trackfileww3-IW-filled-on-thefly.txt"
    start = datetime.datetime(2019, 1, 1)
    stop = datetime.datetime(2025, 1, 1)
    # modify initial listing with 2 more args
    fid = open(listing, "w")
    content = open(listing).readlines()
    taille = len(content)
    # for ll in content:
    for dd in rrule.rrule(rrule.DAILY, dtstart=start, until=stop):
        # ll2 = ll.replace("\n", "") + " " + args.version + " " + args.outputdir + "\n"
        ll2 = dd.strftime("%Y%m%d") + "\n"
        fid.write(ll2)
    fid.close()
    pbs = "/home1/datahome/agrouaze/sources/projet_sarwave/ww3spectra-trackfile-iw/build-trackfile-ww3-for-iw.pbs"
    # call prun
    opts = " --split-max-lines=%s --background -e " % (
        np.ceil(taille / 9900.0).astype(int)
    )  # to respect prun constraint on the number max of sublistings 10000

    cmd = prunexe + opts + pbs + " " + listing
    logging.info("cmd to cast = %s", cmd)
    st = subprocess.check_call(cmd, shell=True)
    logging.info("status cmd = %s", st)
