"""Configuration management using OmegaConf"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env file
load_dotenv()

_config: Optional[DictConfig] = None
_config_path: Optional[Path] = None


def _get_default_config_path() -> Path:
    """Get the default config file path."""
    return Path(__file__).parent.parent / "config" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses config/default.yaml

    Returns:
        OmegaConf DictConfig object
    """
    global _config, _config_path

    if config_path is None:
        config_path = _get_default_config_path()
    else:
        config_path = Path(config_path)

    # Reload if explicit path provided or config not yet loaded
    if _config is None or (_config_path != config_path):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        _config = OmegaConf.load(config_path)
        _config_path = config_path

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


def get_data_path(subdir: str) -> Path:
    """
    Get full path for a data subdirectory.

    Args:
        subdir: One of 'datasets', 'outputs', 'models', 'indices'

    Returns:
        Full path to the subdirectory
    """
    import os

    config = get_config()
    data_dir = Path(config.paths.data_dir)

    # If path is already absolute, use it as-is
    if data_dir.is_absolute():
        return data_dir / config.paths.subdirs[subdir]

    # Resolve relative paths based on environment
    # Modal sets MODAL_IMAGE_ID when running in container
    is_modal = os.getenv("MODAL_IMAGE_ID") is not None

    if is_modal:
        # Running on Modal - resolve relative to /root/searchlm
        project_root = Path("/root/searchlm")
    else:
        # Running locally - resolve relative to project root
        project_root = Path(__file__).parent.parent

    return project_root / data_dir / config.paths.subdirs[subdir]
