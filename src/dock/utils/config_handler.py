"""
Shared config resolution for MCP and FastAPI.

Provides a unified interface to validate and resolve training config
to the format expected by dock.start(). Designed for future extension
(e.g. FastAPI UploadFile → parse → dict).
"""

import json
import yaml
from typing import Dict, Any


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and return config dict for dock.start().

    Used by MCP and FastAPI when config is provided as a dict (JSON body).

    Args:
        config: LlamaFactory config as dictionary.

    Returns:
        Validated config dict (pass-through if valid).

    Raises:
        ValueError: If config is not a valid non-empty dict.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary (JSON object)")
    if not config:
        raise ValueError("config cannot be empty")
    return config


def parse_config_content(content: str) -> Dict[str, Any]:
    """
    Parse YAML or JSON string to config dict.

    For future use: FastAPI UploadFile → read content → parse_config_content().

    Args:
        content: Raw YAML or JSON string (e.g. from file content).

    Returns:
        Parsed config dict.

    Raises:
        ValueError: If content is neither valid YAML nor JSON.

    Example:
        >>> content = (await upload_file.read()).decode('utf-8')
        >>> config = parse_config_content(content)
    """
    content = content.strip()
    if not content:
        raise ValueError("config content cannot be empty")

    # Try YAML first (more permissive)
    try:
        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict):
            return parsed
    except yaml.YAMLError:
        pass

    # Fallback to JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse config: {e}") from e

    raise ValueError("config must parse to a dictionary")


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base. Override values overwrite same keys in base.
    Nested dicts are merged recursively (override wins for leaf values).

    Used when both UploadFile and config (JSON) are provided: file as base, config as overrides.

    Args:
        base: Base config (e.g. from parsed file).
        override: Override config (e.g. from JSON body). Same keys overwrite base.

    Returns:
        Merged config dict.

    Example:
        >>> base = {"a": 1, "b": {"x": 10}}
        >>> override = {"b": {"x": 20, "y": 30}}
        >>> merge_config(base, override)
        {"a": 1, "b": {"x": 20, "y": 30}}
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result
