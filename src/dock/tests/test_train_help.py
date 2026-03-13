"""
Unit tests for LlamaFactoryDock.get_train_help() and related cache logic.

Tests cover:
- In-memory cache (layer 1)
- File cache (layer 2)
- Cache miss and container invocation
- prefetch_train_help
- File cache helpers (atomic write, roundtrip)
- DryRun mode

All Docker interactions are mocked — no running Docker daemon is required.

Usage:
    pytest src/dock/tests/test_train_help.py -v
"""

import os
import pathlib

import docker
import pytest
from unittest.mock import MagicMock, patch

from dock.dock import instance as inst_module


def _make_dock(mock_from_env):
    """Create a LlamaFactoryDock with a mocked docker client."""
    mock_client = MagicMock()
    mock_client.errors = docker.errors
    mock_from_env.return_value = mock_client
    dock = inst_module.LlamaFactoryDock()
    return dock, mock_client


# ---------------------------------------------------------------------------
# Layer-1: in-memory cache
# ---------------------------------------------------------------------------

class TestGetTrainHelpMemoryCache:
    """Tests for in-memory cache layer."""

    @patch("dock.dock.instance.docker.from_env")
    def test_returns_memory_cache_when_digest_matches(self, mock_from_env, patched_cache_dir):
        dock, mock_client = _make_dock(mock_from_env)

        digest = "sha256:deadbeef"
        dock._train_help_cache_key = digest
        dock._train_help_cache = "cached help text"

        mock_image = MagicMock()
        mock_image.id = digest
        mock_client.images.get.return_value = mock_image

        result = dock.get_train_help()

        assert result == "cached help text"
        mock_client.containers.run.assert_not_called()

    @patch("dock.dock.instance.docker.from_env")
    def test_bypasses_memory_cache_when_digest_changes(self, mock_from_env, patched_cache_dir):
        """Stale in-memory cache (old digest) must NOT be served."""
        dock, mock_client = _make_dock(mock_from_env)

        dock._train_help_cache_key = "sha256:old"
        dock._train_help_cache = "stale help"

        new_digest = "sha256:new"
        mock_image = MagicMock()
        mock_image.id = new_digest
        mock_client.images.get.return_value = mock_image
        mock_client.containers.run.return_value = b"fresh help"

        result = dock.get_train_help()

        assert result == "fresh help"
        mock_client.containers.run.assert_called_once()
        assert dock._train_help_cache_key == new_digest


# ---------------------------------------------------------------------------
# Layer-2: file cache
# ---------------------------------------------------------------------------

class TestGetTrainHelpFileCache:
    """Tests for file cache layer."""

    @patch("dock.dock.instance.docker.from_env")
    def test_returns_file_cache_and_populates_memory(self, mock_from_env, patched_cache_dir):
        dock, mock_client = _make_dock(mock_from_env)

        digest = "sha256:abc123"
        mock_image = MagicMock()
        mock_image.id = digest
        mock_client.images.get.return_value = mock_image

        # Pre-populate file cache
        cache_file = patched_cache_dir / f"train-help-{digest.replace(':', '-')[:20]}.txt"
        cache_file.write_text("file-cached help", encoding="utf-8")

        result = dock.get_train_help()

        assert result == "file-cached help"
        mock_client.containers.run.assert_not_called()
        assert dock._train_help_cache == "file-cached help"
        assert dock._train_help_cache_key == digest

    @patch("dock.dock.instance.docker.from_env")
    def test_ignores_file_cache_for_different_digest(self, mock_from_env, patched_cache_dir):
        """File cache for a different digest must not be served."""
        dock, mock_client = _make_dock(mock_from_env)

        old_digest = "sha256:old"
        new_digest = "sha256:filetest-new"
        mock_image = MagicMock()
        mock_image.id = new_digest
        mock_client.images.get.return_value = mock_image

        old_cache_file = patched_cache_dir / f"train-help-{old_digest.replace(':', '-')[:20]}.txt"
        old_cache_file.write_text("old help", encoding="utf-8")

        mock_client.containers.run.return_value = b"new help"

        result = dock.get_train_help()

        assert result == "new help"
        mock_client.containers.run.assert_called_once()


# ---------------------------------------------------------------------------
# Full cache miss
# ---------------------------------------------------------------------------

class TestGetTrainHelpCacheMiss:
    """Tests for cache miss and container invocation."""

    @patch("dock.dock.instance.docker.from_env")
    def test_runs_container_and_writes_caches(self, mock_from_env, patched_cache_dir):
        dock, mock_client = _make_dock(mock_from_env)

        digest = "sha256:fresh"
        mock_image = MagicMock()
        mock_image.id = digest
        mock_client.images.get.return_value = mock_image
        mock_client.containers.run.return_value = b"help output"

        result = dock.get_train_help()

        assert result == "help output"
        mock_client.containers.run.assert_called_once_with(
            image=dock.docker_image,
            platform=dock.docker_container_platform,
            command=["llamafactory-cli", "train", "--help"],
            remove=True,
            stdout=True,
            stderr=True,
        )
        assert dock._train_help_cache == "help output"
        assert dock._train_help_cache_key == digest

        cache_file = patched_cache_dir / f"train-help-{digest.replace(':', '-')[:20]}.txt"
        assert cache_file.exists()
        assert cache_file.read_text(encoding="utf-8") == "help output"

    @patch("dock.dock.instance.docker.from_env")
    def test_second_call_uses_memory_cache(self, mock_from_env, patched_cache_dir):
        """After a cache miss, the second call must hit memory without re-running the container."""
        dock, mock_client = _make_dock(mock_from_env)

        digest = "sha256:repeated"
        mock_image = MagicMock()
        mock_image.id = digest
        mock_client.images.get.return_value = mock_image
        mock_client.containers.run.return_value = b"help text"

        dock.get_train_help()
        dock.get_train_help()

        assert mock_client.containers.run.call_count == 1


# ---------------------------------------------------------------------------
# prefetch_train_help
# ---------------------------------------------------------------------------

class TestPrefetchTrainHelp:
    """Tests for prefetch_train_help."""

    @patch("dock.dock.instance.docker.from_env")
    def test_returns_result_on_success(self, mock_from_env, patched_cache_dir):
        dock, mock_client = _make_dock(mock_from_env)

        digest = "sha256:ok"
        mock_image = MagicMock()
        mock_image.id = digest
        mock_client.images.get.return_value = mock_image
        mock_client.containers.run.return_value = b"help"

        result = dock.prefetch_train_help()

        assert result == "help"

    @patch("dock.dock.instance.docker.from_env")
    def test_returns_none_on_failure(self, mock_from_env, patched_cache_dir):
        dock, mock_client = _make_dock(mock_from_env)

        mock_client.images.get.side_effect = Exception("Docker not available")

        result = dock.prefetch_train_help()

        assert result is None


# ---------------------------------------------------------------------------
# File-cache helper methods
# ---------------------------------------------------------------------------

class TestFileCacheHelpers:
    """Tests for file cache helper methods."""

    @patch("dock.dock.instance.docker.from_env")
    def test_cache_file_path_format(self, mock_from_env, patched_cache_dir):
        dock, _ = _make_dock(mock_from_env)

        digest = "sha256:abcdef1234567890"
        path = dock._get_train_help_cache_file(digest)

        assert "train-help-" in path.name
        assert path.name.endswith(".txt")
        assert ":" not in path.name

    @patch("dock.dock.instance.docker.from_env")
    def test_read_returns_none_when_file_missing(self, mock_from_env, patched_cache_dir):
        dock, _ = _make_dock(mock_from_env)

        result = dock._read_train_help_file_cache("sha256:nonexistent")

        assert result is None

    @patch("dock.dock.instance.docker.from_env")
    def test_write_then_read_roundtrip(self, mock_from_env, patched_cache_dir):
        dock, _ = _make_dock(mock_from_env)

        digest = "sha256:roundtrip"
        content = "some help content\nwith multiple lines"

        dock._write_train_help_file_cache(digest, content)
        result = dock._read_train_help_file_cache(digest)

        assert result == content

    @patch("dock.dock.instance.docker.from_env")
    def test_write_is_atomic(self, mock_from_env, patched_cache_dir):
        """Write must go through a .tmp file then os.replace."""
        dock, _ = _make_dock(mock_from_env)

        with patch("os.replace", wraps=os.replace) as spy_replace:
            dock._write_train_help_file_cache("sha256:atomic", "content")
            spy_replace.assert_called_once()
            src, _ = spy_replace.call_args.args
            assert str(src).endswith(".tmp")

    @patch("dock.dock.instance.docker.from_env")
    def test_write_silently_skips_on_permission_error(self, mock_from_env, patched_cache_dir):
        """Write failure must not raise."""
        dock, _ = _make_dock(mock_from_env)

        with patch("os.replace", side_effect=PermissionError("no write")):
            try:
                dock._write_train_help_file_cache("sha256:fail", "content")
            except Exception as exc:
                pytest.fail(f"_write_train_help_file_cache raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# DryRun mode
# ---------------------------------------------------------------------------

class TestDryRunDock:
    """Tests for LlamaFactoryDryRunDock.get_train_help."""

    @patch("dock.dock.instance.docker.from_env")
    def test_get_train_help_returns_stub(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        from dock.dock.dryrun import LlamaFactoryDryRunDock

        dock = LlamaFactoryDryRunDock()
        result = dock.get_train_help()

        assert "DRYRUN" in result
        mock_client.containers.run.assert_not_called()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from .base import run_tests_with_report

    sys.exit(run_tests_with_report(__file__, "train_help"))
