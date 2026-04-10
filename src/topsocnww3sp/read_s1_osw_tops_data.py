import glob
import logging
import os

import numpy as np
import xarray as xr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_osw(group, lst_sar_files_osw, dev=False):
    """

    Args:
        group (str): group in the osw file to read, can be 'intraburst' or 'interburst'
        lst_sar_files_osw (list): list of osw files to read
        dev (bool): flag to indicate if the script is run in development mode, if True, only a subset of files will be processed for testing purposes

    Returns:
        fat_osw (xarray dataset): dataset concatenated along the tiles dimension and reduced along the oswKyBinSize dimension to the minimum common size across all files
        coords_osw (dict): dictionary with the coordinates of the osw data, with keys 'lon_osw' and 'lat_osw'


    """
    cpt_subswath_concat = 0
    cpt_subswath_discard = 0
    # lst_sar_files_osw = glob.glob(os.path.join(dd_sar,'dataset_'+dataset_chosen,'*SAFE','measurement','*osw*nc'))
    logger.info("Found %d osw files " % (len(lst_sar_files_osw)))
    coords_osw = {}
    coords_osw["lon_osw"] = []
    coords_osw["lat_osw"] = []
    fat_osw = None
    concatosw = []
    logger.info("Processing group %s ))", group)
    min_kybinsize = 200
    if dev:
        lst_sar_files_osw = lst_sar_files_osw[:2]  # for testing with a subset of files
        logger.info("Running in development mode, only processing the first 2 files")
    for ii in tqdm(range(len(lst_sar_files_osw))):
        onesarfileocn = lst_sar_files_osw[ii]
        dsosw = xr.open_dataset(onesarfileocn, group=group)
        dsosw["onesarfileocn"] = onesarfileocn
        coords_osw["lon_osw"] = np.hstack(
            [coords_osw["lon_osw"], dsosw["oswLon"].squeeze().values.ravel()]
        )
        coords_osw["lat_osw"] = np.hstack(
            [coords_osw["lat_osw"], dsosw["oswLat"].squeeze().values.ravel()]
        )
        # dsosw = dsosw.isel({'oswKyBinSize':slice(0,90)})

        min_kybinsize = min(min_kybinsize, dsosw["oswKyBinSize"].size)
        # drop tiles over land.
        # dsosw = dsosw.where(dsosw['oswLandFlag']==0,drop=True)
        # reduce the oswKy vector to [0:-2] sicne it appears that for some IW intraburst tiles the 2 last bins are NaN.
        # dsosw = dsosw.isel({'oswKyBinSize':slice(0,-2)})
        # dsosw = dsosw.isel({'oswKyBinSize':slice(0,90)}) # with IPF403  cannot reindex or align along dimension 'oswKyBinSize' because of conflicting dimension sizes: {67, 63}
        if (
            dsosw["oswLandFlag"].values == 0
        ).any():  # it means at least one tile is over ocean
            if np.isnan(dsosw.oswKy.values).any():
                pass  # in the end, the issue with nans in oswKy is solved at the plot step when a given tile is selected.
                # could be a problem is the tile of the whole subswath is on land
                # breakpoint()
                # print('dsosw.oswKy.values',dsosw.oswKy.values)
            concatosw.append(dsosw.stack(tiles=("oswRaSize", "oswAzSize")))
            cpt_subswath_concat += 1
        else:
            cpt_subswath_discard += 1
    logger.info(
        "Number of subswath concatenated: %d, discarded because only land: %d",
        cpt_subswath_concat,
        cpt_subswath_discard,
    )
    logger.info("min_kybinsize %d", min_kybinsize)
    # reduce the oswKyBinSize dimension to the minimum common size across all files, otherwise cannot concatenate along tiles dimension because of different sizes of oswKyBinSize across files (with IPF403)
    concatosw2 = []
    for oo in concatosw:
        concatosw2.append(oo.isel({"oswKyBinSize": slice(0, min_kybinsize)}))
        # concatosw2.append(oo.reindex(oswKyBinSize=np.arange(min_kybinsize)))
    fat_osw = xr.concat(concatosw2, dim="subswath", join="outer")
    return fat_osw, coords_osw


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read S1 OSW tops data and concatenate along tiles dimension"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="intraburst",
        help="Group in the osw file to read, can be intraburst or interburst",
    )
    parser.add_argument(
        "--path_to_sar_files",
        type=str,
        required=True,
        help="Path to the directory containing the SAR files with osw data, e.g. /path/to/S1/files/*SAFE/measurement/*osw*nc",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="INFO",
        help="Logging level, e.g. INFO, DEBUG",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Flag to indicate if the script is run in development mode (for testing with a subset of files)",
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.logging_level.upper(), None))

    lst_sar_files_osw = glob.glob(
        os.path.join(args.path_to_sar_files, "*SAFE", "measurement", "*osw*nc")
    )
    logger.info("Found %d osw files " % (len(lst_sar_files_osw)))
    fat_osw, coords_osw = read_osw(args.group, lst_sar_files_osw, dev=args.dev)
    logger.info(
        "Done reading osw files, fat_osw shape: %s", str(coords_osw["lon_osw"].shape)
    )
    logger.info(" osw data : %s", fat_osw)
