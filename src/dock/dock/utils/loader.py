import json
import yaml
import pathlib
from typing import Any, List, Union

class ConfigLoader:

    @staticmethod
    def load(source_file: Union[None, str, pathlib.Path] = None) -> List[Any]:
        """
        Load tool configurations from YAML or JSON file.

        Args:
            source_file: Path to configuration file (.yaml, .yml, or .json)

        Returns:
            List of tool configurations from the 'tools' key in the file.
            Returns empty list if source_file is None or 'tools' key not found.

        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML file is invalid
            json.JSONDecodeError: If JSON file is invalid
        """
        if not source_file:
            return []

        if isinstance(source_file, str):
            source_file = pathlib.Path(source_file).resolve()

        if not source_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {source_file}")

        if source_file.suffix in {".yaml", ".yml"}:
            return ConfigLoader.load_from_yaml(source_file)

        elif source_file.suffix == ".json":
            return ConfigLoader.load_from_json(source_file)

        else:
            raise ValueError(
                f"Unsupported file extension: {source_file.suffix}. "
                f"Supported extensions: .yaml, .yml, .json"
            )

    @staticmethod
    def load_from_yaml(source_file: pathlib.Path) -> List[Any]:
        """
        Load tool configurations from YAML file.

        Args:
            source_file: Path to YAML file

        Returns:
            List of tool configurations from the 'tools' key, or empty list if not found
        """
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (yaml.YAMLError, ValueError):
            # Handle empty or invalid YAML files
            return []

    @staticmethod
    def load_from_json(source_file: pathlib.Path) -> List[Any]:
        """
        Load tool configurations from JSON file.

        Args:
            source_file: Path to JSON file

        Returns:
            List of tool configurations from the 'tools' key, or empty list if not found
        """
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except (json.JSONDecodeError, ValueError):
            # Handle empty or invalid JSON files
            return []