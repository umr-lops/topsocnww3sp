#!/usr/bin/env python3
"""Utility functions for the topsocnww3sp package."""

import logging
from pathlib import Path

import numpy as np
import yaml


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
