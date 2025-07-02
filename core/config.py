# core/config.py

import os
import yaml

def load_config(path: str = None) -> dict:
    """
    Load the YAML config (assumed UTF-8) from core/config.yaml by default.
    """
    if path is None:
        # locate config.yaml in the same folder as this file
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
