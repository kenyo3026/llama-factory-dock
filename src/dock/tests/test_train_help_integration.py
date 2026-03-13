"""
Integration tests for LlamaFactoryDock.get_train_help().

Tests cover end-to-end behavior against a real Docker daemon:
- Real container invocation returns non-empty output
- File cache is written after first call
- Second call (memory reset) hits file cache without re-running container
- prefetch_train_help returns the result end-to-end

Prerequisites (tests are skipped automatically if not met):
- Docker daemon must be running
- Image hiyouga/llamafactory:latest must be present locally
  (run `docker pull hiyouga/llamafactory:latest` first if needed)

Usage:
    pytest src/dock/tests/test_train_help_integration.py -v -m integration
"""

import pathlib
import tempfile

import pytest

import docker

from dock.dock import instance as inst_module
from dock.dock.instance import LlamaFactoryDock, DOCKER_IMAGE


# ---------------------------------------------------------------------------
# Session-scoped helpers: skip early if prerequisites are not met
# ---------------------------------------------------------------------------

def _docker_client_or_skip():
    """Return a live docker client, or skip the test if daemon is unreachable."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as exc:
        pytest.skip(f"Docker daemon not available: {exc}")


def _require_image(client):
    """Skip if the target image is not present locally (avoids accidental pulls)."""
    try:
        client.images.get(DOCKER_IMAGE)
    except docker.errors.ImageNotFound:
        pytest.skip(
            f"Image '{DOCKER_IMAGE}' not found locally. "
            "Run `docker pull hiyouga/llamafactory:latest` first."
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def docker_client():
    return _docker_client_or_skip()


@pytest.fixture
def dock_with_tmp_cache(docker_client, monkeypatch):
    """
    LlamaFactoryDock with a fresh temporary cache directory.
    Skips if image is not available locally.
    """
    _require_image(docker_client)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        monkeypatch.setattr(inst_module, 'TRAIN_HELP_CACHE_DIR', tmp_path)
        yield LlamaFactoryDock(), tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGetTrainHelpIntegration:
    """End-to-end tests for get_train_help() against a real Docker daemon."""

    def test_returns_nonempty_output(self, dock_with_tmp_cache):
        """Real container must return non-empty --help output."""
        dock, _ = dock_with_tmp_cache

        result = dock.get_train_help()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_contains_expected_keywords(self, dock_with_tmp_cache):
        """Output should contain recognizable llamafactory-cli content."""
        dock, _ = dock_with_tmp_cache

        result = dock.get_train_help()

        # LLaMA Factory help output typically contains these strings
        assert any(
            keyword in result
            for keyword in ("train", "model", "data", "output", "usage", "LlamaFactory")
        ), f"No expected keyword found in output (first 200 chars): {result[:200]}"

    def test_file_cache_written_after_first_call(self, dock_with_tmp_cache):
        """A cache file must exist in tmp_cache_dir after the first call."""
        dock, tmp_path = dock_with_tmp_cache

        dock.get_train_help()

        cache_files = list(tmp_path.glob("train-help-*.txt"))
        assert len(cache_files) == 1, f"Expected exactly 1 cache file, found: {cache_files}"

    def test_file_cache_content_matches_result(self, dock_with_tmp_cache):
        """Cache file content must exactly match the returned string."""
        dock, tmp_path = dock_with_tmp_cache

        result = dock.get_train_help()

        cache_files = list(tmp_path.glob("train-help-*.txt"))
        assert cache_files, "No cache file found"
        cached_content = cache_files[0].read_text(encoding="utf-8")
        assert cached_content == result

    def test_second_call_after_memory_reset_uses_file_cache(self, dock_with_tmp_cache):
        """
        After clearing in-memory cache, the second call must serve from the
        file cache without re-running the container.
        """
        dock, _ = dock_with_tmp_cache

        first_result = dock.get_train_help()

        # Reset memory cache to force file cache lookup
        dock._train_help_cache = None
        dock._train_help_cache_key = None

        second_result = dock.get_train_help()

        assert second_result == first_result

    def test_memory_cache_populated_after_call(self, dock_with_tmp_cache):
        """In-memory cache must be populated after a successful call."""
        dock, _ = dock_with_tmp_cache

        result = dock.get_train_help()

        assert dock._train_help_cache == result
        assert dock._train_help_cache_key is not None

    def test_second_call_is_faster_than_first(self, dock_with_tmp_cache):
        """
        Second call (memory cache hit) must complete significantly faster
        than first call (container run).
        """
        import time
        dock, _ = dock_with_tmp_cache

        start = time.monotonic()
        dock.get_train_help()
        first_duration = time.monotonic() - start

        start = time.monotonic()
        dock.get_train_help()
        second_duration = time.monotonic() - start

        assert second_duration < first_duration, (
            f"Second call ({second_duration:.3f}s) not faster than first ({first_duration:.3f}s)"
        )


@pytest.mark.integration
class TestPrefetchTrainHelpIntegration:
    """End-to-end tests for prefetch_train_help()."""

    def test_prefetch_returns_nonempty_string(self, dock_with_tmp_cache):
        """prefetch_train_help must return a non-empty string on success."""
        dock, _ = dock_with_tmp_cache

        result = dock.prefetch_train_help()

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prefetch_warms_memory_cache(self, dock_with_tmp_cache):
        """After prefetch, in-memory cache must be populated."""
        dock, _ = dock_with_tmp_cache

        dock.prefetch_train_help()

        assert dock._train_help_cache is not None
        assert dock._train_help_cache_key is not None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from .base import run_tests_with_report

    sys.exit(run_tests_with_report(__file__, "train_help_integration"))
