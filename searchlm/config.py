"""Configuration management using OmegaConf"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env file
load_dotenv()

_config: Optional[DictConfig] = None


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses config/default.yaml

    Returns:
        OmegaConf DictConfig object
    """
    global _config

    if _config is not None:
        return _config

    if config_path is None:
        # Default to config/default.yaml relative to project root
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _config = OmegaConf.load(config_path)
    return _config


def get_config() -> DictConfig:
    """Get the loaded configuration (loads default if not loaded yet)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def merge_config(overrides: dict) -> DictConfig:
    """
    Merge overrides into the configuration.

    Args:
        overrides: Dictionary of configuration overrides

    Returns:
        Updated configuration
    """
    config = get_config()
    override_conf = OmegaConf.create(overrides)
    return OmegaConf.merge(config, override_conf)
