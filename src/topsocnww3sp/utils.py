#!/usr/bin/env python3
"""Utility functions for the topsocnww3sp package."""

from pathlib import Path

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

    with config_path.open() as f:
        return yaml.safe_load(f)
