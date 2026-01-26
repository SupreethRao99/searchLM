"""
Logging utilities for consistent output formatting.

This module provides helper functions for consistent logging
and output formatting across the application.
"""


def print_header(text: str, width: int = 60, char: str = "="):
    """
    Print a formatted header with a border.

    Args:
        text: Header text to display
        width: Width of the header in characters
        char: Character to use for the border
    """
    print(char * width)
    print(text)
    print(char * width)


def print_section(text: str, width: int = 60, char: str = "-"):
    """
    Print a formatted section divider.

    Args:
        text: Section text to display
        width: Width of the divider in characters
        char: Character to use for the divider
    """
    print("\n" + char * width)
    print(text)
    print(char * width + "\n")


def format_metric(name: str, value: float, precision: int = 4) -> str:
    """
    Format a metric name and value for display.

    Args:
        name: Metric name
        value: Metric value
        precision: Number of decimal places

    Returns:
        Formatted metric string
    """
    return f"{name}: {value:.{precision}f}"


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """
    Print a simple progress indicator.

    Args:
        current: Current progress count
        total: Total count
        prefix: Prefix text for the progress message
    """
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
