"""Shared validation helpers for deterministic tourism tools."""

import math


def validate_required_string(value: object, field_name: str) -> str:
    """Return a stripped required string or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and cannot be empty.")
    return value.strip()


def validate_optional_string(value: object, field_name: str) -> str:
    """Return a stripped optional string or raise ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value.strip()


def validate_positive_integer(value: object, field_name: str) -> int:
    """Return a positive integer or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def validate_non_negative_number(value: object, field_name: str) -> float:
    """Return a finite non-negative number or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative number.")

    normalized_value = float(value)
    if not math.isfinite(normalized_value) or normalized_value < 0:
        raise ValueError(f"{field_name} must be a non-negative number.")
    return normalized_value
