import os
import yaml
from pathlib import Path

# Base directory of the project (Yolo_image_search root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Config directory
CONFIG_DIR = BASE_DIR / "configs"

# Ensure directories exist so the app doesn't crash on first run
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Add the missing load_config function
def load_config(config_name="default.yaml"):
    """
    Loads a YAML configuration file from the configs directory.
    """
    config_path = CONFIG_DIR / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as file:
        return yaml.safe_load(file)