#!/usr/bin/env python3
"""Read S1 OSW tops data and concatenate along tiles dimension."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_source_path(full_path: str) -> str:
    """Extract the meaningful part of the OSW file path."""
    path = Path(full_path)
    parts = path.parts
    # Chercher l'index du dossier "dataset_G" (ou un motif connu)
    try:
        idx = parts.index("dataset_G")  # ou "dataset_validation_xspecTops_ipf403" ?
        # Prendre tout après cet index
        clean = Path(*parts[idx + 1 :])  # +1 pour sauter dataset_G
    except ValueError:
        # Fallback : prendre les 3 derniers dossiers (SAFE/measurement/fichier)
        clean = Path(*parts[-3:]) if len(parts) >= 3 else Path(path.name)
    return str(clean)


def read_osw(
    group: str, lst_sar_files_osw: list[Path], dev: bool = False
) -> tuple[xr.Dataset, dict[str, np.ndarray]]:
    """
    Read OSW files and concatenate along tiles dimension.

    Args:
        group (str): group in the osw file to read, can be 'intraburst' or 'interburst'
        lst_sar_files_osw (list[Path]): list of osw files to read
        dev (bool): flag to indicate if the script is run in development mode,
                    if True, only a subset of files will be processed for testing purposes

    Returns:
        fat_osw (xarray dataset): dataset concatenated along the tiles dimension
            and reduced along oswKyBinSize to minimum common size
        coords_osw (dict): dictionary with keys 'lon_osw' and 'lat_osw'
    """
    cpt_subswath_concat = 0
    cpt_subswath_discard = 0
    # Use lazy formatting for logging
    logger.info("Found %d osw files ", len(lst_sar_files_osw))
    coords_osw: dict[str, np.ndarray] = {}
    coords_osw["lon_osw"] = []
    coords_osw["lat_osw"] = []
    fat_osw = None
    concatosw = []
    logger.debug("Processing group %s ", group)
    min_kybinsize = 200
    if dev:
        lst_sar_files_osw = lst_sar_files_osw[:2]
        logger.info("Running in development mode, only processing the first 2 files")
    for ii in tqdm(range(len(lst_sar_files_osw))):
        onesarfileocn = lst_sar_files_osw[ii]
        dsosw = xr.open_dataset(onesarfileocn, group=group)
        # dsosw["onesarfileocn"] = onesarfileocn
        dsosw.attrs["source_file"] = clean_source_path(onesarfileocn)
        # Use .to_numpy() instead of .values
        coords_osw["lon_osw"] = np.hstack(
            [coords_osw["lon_osw"], dsosw["oswLon"].squeeze().to_numpy().ravel()]
        )
        coords_osw["lat_osw"] = np.hstack(
            [coords_osw["lat_osw"], dsosw["oswLat"].squeeze().to_numpy().ravel()]
        )

        min_kybinsize = min(min_kybinsize, dsosw["oswKyBinSize"].size)

        if (
            dsosw["oswLandFlag"].to_numpy() == 0
        ).any():  # at least one tile is over ocean
            if np.isnan(dsosw.oswKy.to_numpy()).any():
                pass
            concatosw.append(dsosw.stack(tiles=("oswRaSize", "oswAzSize")))
            cpt_subswath_concat += 1
        else:
            cpt_subswath_discard += 1
    logger.info(
        "Number of subswath concatenated: %d, discarded because only land: %d",
        cpt_subswath_concat,
        cpt_subswath_discard,
    )
    logger.debug("min_kybinsize %d", min_kybinsize)
    # Use list comprehension instead of for loop (PERF401)
    concatosw2 = [
        oo.isel({"oswKyBinSize": slice(0, min_kybinsize)}) for oo in concatosw
    ]
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

    logger.setLevel(getattr(logging, args.logging_level.upper(), "INFO"))

    lst_sar_files_osw_from_main = list(Path(args.path_to_sar_files).rglob("**/*osw*nc"))

    logger.info("Found %d osw files ", len(lst_sar_files_osw_from_main))
    fat_osw_m, coords_osw_m = read_osw(
        args.group, lst_sar_files_osw_from_main, dev=args.dev
    )
    logger.info(
        "Done reading osw files, fat_osw shape: %s", str(coords_osw_m["lon_osw"].shape)
    )
    logger.info(" osw data : %s", fat_osw_m)
