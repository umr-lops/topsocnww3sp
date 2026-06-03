#!/usr/bin/env python3
"""Utility functions for the topsocnww3sp package."""

import logging
from pathlib import Path

import numpy as np
import yaml
import xarray as xr
from datetime import datetime, timedelta, timezone
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


def get_config(path_config: str | Path | None = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        path_config (str or Path, optional): Path to the YAML configuration file.
            If None, defaults to 'config.yml' in the same directory as this script.

    Returns:
        dict: Configuration parameters loaded from the YAML file.
    """
    if path_config is None:
        config_path = Path(__file__).parent / "config.yml"
    else:
        config_path = Path(path_config)

    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    Args:
        lon1: Longitude of first point in decimal degrees
        lat1: Latitude of first point in decimal degrees
        lon2: Longitude of second point in decimal degrees
        lat2: Latitude of second point in decimal degrees

    Returns:
        Distance between the two points in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def format_logs(logger: logging.Logger, level: str) -> logging.Logger:
    """
    Configure the logger with a specific format and level.

    Args:
        logger: The logger instance to configure.
        level: The logging level as a string (e.g., "debug", "info").
    returns:
        The configured logger instance.

    """
    # Logging config
    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    # level = logging.DEBUG if  else logging.INFO
    # logging.basicConfig(
    #     level=level, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
    # )
    logger.setLevel(logging.DEBUG if level == "debug" else logging.INFO)
    # set formatter
    formatter = logging.Formatter(fmt, datefmt="%d/%m/%Y %H:%S")
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger




def load_ww3_multi_grid(
    ww3_dir: Path, sar_start: datetime, config: dict[str, Any]
) -> xr.Dataset:
    """Load and merge WW3 data from multiple grids (IRI configuration).

    Args:
        ww3_dir: Directory containing WW3 grid subdirectories
        sar_start: SAR acquisition time for temporal filtering
        config: Configuration dictionary with ww3_grids patterns

    Returns:
        Merged xarray.Dataset with all available WW3 grids
    """
    if "ww3_grids" not in config:
        raise ValueError("Missing 'ww3_grids' configuration for multi-grid mode")

    t_thresh = timedelta(minutes=config["TIME_THRESHOLD_MINUTES"])
    t_sar = pd.Timestamp(sar_start).tz_localize(None)
    datasets = []
    grid_names_used = []

    # Build flag meaning string once
    flag_meaning_str = ", ".join([name for name in config["ww3_grids"].keys()])

    for grid_name, grid_config in config["ww3_grids"].items():
        # Build pattern with year
        year_str = sar_start.strftime("%Y")
        pattern = grid_config["pattern"].replace("YYYY", year_str)
        logger.debug("Looking for WW3 files in %s with pattern %s", ww3_dir, pattern)
        files = list(ww3_dir.glob(pattern))

        if not files:
            logger.debug("No files found for grid %s with pattern %s", grid_name, pattern)
            continue

        # Load and concatenate files for this grid
        ds_grid = xr.open_mfdataset(files, combine="nested", concat_dim="time")
        ww3_times = pd.to_datetime(ds_grid.time.to_numpy())

        # Temporal filter
        t_mask = (ww3_times >= t_sar - t_thresh) & (ww3_times <= t_sar + t_thresh)
        if not np.any(t_mask):
            logger.debug("No temporal match for grid %s", grid_name)
            continue

        ds_filtered = ds_grid.isel(time=t_mask)

        # Add provenance variable with integer code for each grid
        # Use an integer code (0, 1, 2, ...) for each grid
        grid_code = list(config["ww3_grids"].keys()).index(grid_name)
        explicite_comment = "0=arctic, 1=antarctic, 2=midlatitude" 
        provenance = xr.DataArray(
            np.full(ds_filtered.sizes["time"], grid_code),
            dims="time",
            attrs={
                "long_name": "WW3 grid provenance",
                "flag_meanings": flag_meaning_str,
                "flag_values": " ".join([str(i) for i in range(len(config["ww3_grids"]))]),
                "comment": explicite_comment,
            },
        )
        ds_filtered["ww3_grid_provenance"] = provenance
        datasets.append(ds_filtered)
        grid_names_used.append(grid_name)

    if not datasets:
        raise FileNotFoundError(f"No WW3 data found in {ww3_dir} for time window around {sar_start}")

    # Merge all grids along time dimension
    ds_merged = xr.concat(datasets, dim="time", join="inner")
    logger.info("Merged %d WW3 grids, total %d time steps", len(datasets), ds_merged.sizes["time"])

    # Add metadata about grids used
    ds_merged.attrs["ww3_grids_merged"] = ", ".join(grid_names_used)

    return ds_merged