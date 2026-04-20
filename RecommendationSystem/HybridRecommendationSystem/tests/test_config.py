import pytest
from src.config import config

def test_config_attributes():
    # Check essential config attributes
    assert config.catalog_name is not None
    assert config.schema_name is not None
    assert config.model_name is not None

def test_paths_resolved():
    # Paths should be resolved correctly
    assert hasattr(config, "data_path")
    assert hasattr(config, "model_path")
    assert config.data_path.endswith("data")
    assert config.model_path.endswith("models")
