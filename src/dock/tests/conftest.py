"""
Pytest configuration and fixtures for dock tests.
"""

import pathlib
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires running Docker daemon and image)",
    )


@pytest.fixture
def tmp_cache_dir():
    """Temporary directory for TRAIN_HELP_CACHE_DIR in tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield pathlib.Path(tmp)


@pytest.fixture
def patched_cache_dir(tmp_cache_dir, monkeypatch):
    """Patch TRAIN_HELP_CACHE_DIR to tmp_cache_dir for the test."""
    from dock.dock import instance as inst_module
    monkeypatch.setattr(inst_module, 'TRAIN_HELP_CACHE_DIR', tmp_cache_dir)
    return tmp_cache_dir
