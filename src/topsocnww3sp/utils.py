import os

import yaml


def get_config(path_config: str | None = None) -> dict:
    """Load configuration from a YAML file.

    Args:
        path_config (str, optional): Path to the YAML configuration file.
                                     If None, defaults to 'config.yml' in the same directory as this script.
    Returns:
        dict: Configuration parameters loaded from the YAML file.

    """
    if path_config is None:
        path_config = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(path_config) as f:
        config = yaml.safe_load(f)
    return config
