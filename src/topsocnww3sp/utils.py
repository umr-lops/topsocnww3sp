import os
import yaml

def get_config(path_config=None):
    """Load configuration from a YAML file."""
    if path_config is None:
        path_config = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(path_config, "r") as f:
        config = yaml.safe_load(f)
    return config