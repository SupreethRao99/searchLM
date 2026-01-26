"""
Validation utilities for input checking.

This module provides helper functions for validating inputs
and ensuring data consistency.
"""

from typing import Any, Dict, List


def validate_dataset_name(dataset_name: str) -> str:
    """
    Validate and normalize dataset name.

    Args:
        dataset_name: Dataset name to validate

    Returns:
        Normalized dataset name (lowercase)

    Raises:
        ValueError: If dataset name is not recognized
    """
    valid_datasets = ["nfcorpus", "scifact"]
    normalized = dataset_name.lower()

    if normalized not in valid_datasets:
        raise ValueError(
            f"Invalid dataset name: {dataset_name}. "
            f"Valid options: {', '.join(valid_datasets)}"
        )

    return normalized


def validate_split_name(split: str) -> str:
    """
    Validate and normalize split name.

    Args:
        split: Split name to validate

    Returns:
        Normalized split name (lowercase)

    Raises:
        ValueError: If split name is not recognized
    """
    valid_splits = ["train", "dev", "test"]
    normalized = split.lower()

    if normalized not in valid_splits:
        raise ValueError(
            f"Invalid split name: {split}. Valid options: {', '.join(valid_splits)}"
        )

    return normalized


def validate_positive_int(value: int, name: str = "value") -> int:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def validate_dict_keys(
    data: Dict[str, Any], required_keys: List[str], name: str = "dictionary"
) -> None:
    """
    Validate that a dictionary contains all required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of required key names
        name: Name of the dictionary (for error messages)

    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing_keys)}")
