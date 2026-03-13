import os
import json
import uuid
import yaml
import pathlib
import tempfile
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Union

import docker
from .utils.loader import ConfigLoader

TRAIN_HELP_CACHE_DIR: pathlib.Path = pathlib.Path.home() / ".cache" / "llama-factory-dock"


PROJECT_ROOT = pathlib.Path(os.getcwd())
DATASET_DIR: pathlib.Path = PROJECT_ROOT / "datasets"
RECIPE_DIR: pathlib.Path = PROJECT_ROOT / "recipes"
OUTPUT_DIR: pathlib.Path = PROJECT_ROOT / "outputs"

DOCKER_IMAGE: str = "hiyouga/llamafactory:latest"
DOCKER_CONTAINER_PLATFORM: str = "linux/amd64"  # Force x86_64 on ARM64 (slower but works)
DOCKER_CONTAINER_GPU_REQUEST = [docker.types.DeviceRequest(device_ids=['4', '5'], capabilities=[['gpu']])]
DOCKER_CONTAINER_SHM_SIZE: str = "8g"

DOCKER_CONTAINER_ROOT: pathlib.Path = pathlib.Path("/app")
DOCKER_CONTAINER_VOLUME_MAP: dict = {
    str(DATASET_DIR): {"bind": str(DOCKER_CONTAINER_ROOT / "datasets"), "mode": "ro"},
    str(RECIPE_DIR): {"bind": str(DOCKER_CONTAINER_ROOT / "examples"), "mode": "ro"},
    str(OUTPUT_DIR): {"bind": str(DOCKER_CONTAINER_ROOT / "outputs"), "mode": "rw"},
}

# Label for identifying our containers
DOCK_LABEL_KEY = "llama-factory-dock.managed"
DOCK_LABEL_VALUE = "true"
JOB_ID_LABEL_KEY = "llama-factory-dock.job_id"


@dataclass(frozen=True)
class TrainingStatus:
    """Training job status"""
    PENDING: str = "pending"
    PULLING_IMAGE: str = "pulling_image"
    RUNNING: str = "running"
    PAUSED: str = "paused"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    STOPPED: str = "stopped"


@dataclass
class TrainingJob:
    """Training job data model - built from Docker container"""
    job_id: str  # Human-friendly ID (8 chars), can use for querying
    container_id: str  # Docker container ID (short 12 chars or full 64 chars)
    status: str = TrainingStatus.PENDING
    config: Optional[Dict] = None
    metrics: Optional[Dict] = None
    image: Optional[str] = None  # Docker image used (e.g. hiyouga/llamafactory:latest)

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    error_message: Optional[str] = None

    def get_progress_percentage(self) -> float:
        """Calculate training progress percentage"""
        if not self.metrics or not hasattr(self.metrics, 'total_steps'):
            return 0.0
        if not hasattr(self.metrics, 'step'):
            return 0.0
        return (self.metrics.step / self.metrics.total_steps) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "job_id": self.job_id,
            "container_id": self.container_id,
            "status": self.status,
            "config": self.config,
            "metrics": self.metrics,
            "image": self.image,
            "progress_percentage": self.get_progress_percentage(),
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "started_at": self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            "completed_at": self.completed_at.isoformat() if isinstance(self.completed_at, datetime) else self.completed_at,
            "error_message": self.error_message,
        }


class LlamaFactoryDock:

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.docker_client = docker.from_env()
        self.docker_image = DOCKER_IMAGE
        self.docker_container_platform = DOCKER_CONTAINER_PLATFORM
        self.docker_container_root = DOCKER_CONTAINER_ROOT
        self.dataset_dir = DATASET_DIR
        self.recipe_dir = RECIPE_DIR
        self.output_dir = OUTPUT_DIR

        # Create host directories if they don't exist
        for d in (self.dataset_dir, self.recipe_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

        # In-memory jobs during preparation (e.g. PULLING_IMAGE) before container exists
        self._preparing_jobs: Dict[str, TrainingJob] = {}

        # Two-layer cache for llamafactory-cli train --help output
        # Key is the image digest (SHA256), so cache auto-invalidates when image changes
        self._train_help_cache: Optional[str] = None
        self._train_help_cache_key: Optional[str] = None

        self.logger.info(f"LlamaFactoryDock initialized")
        self.logger.debug(f"Docker image: {self.docker_image}")
        self.logger.debug(f"Platform: {self.docker_container_platform}")
        self.logger.debug(f"Dataset directory: {self.dataset_dir}")
        self.logger.debug(f"Recipe directory: {self.recipe_dir}")
        self.logger.debug(f"Output directory: {self.output_dir}")

    def start(
        self,
        config: Union[None, str, pathlib.Path] = None,
        auto_pull: bool = True,
        **override_config,
    ) -> TrainingJob:
        """Start a training job"""
        self.logger.info(f"Starting training job with config type: {type(config).__name__}")

        if isinstance(config, str):
            config = ConfigLoader.load(config)

        created_at = datetime.now()
        job_id = uuid.uuid4().hex[:8]  # Human-friendly 8-char ID

        try:
            if auto_pull:
                preparing_job = TrainingJob(
                    job_id=job_id,
                    container_id="",
                    status=TrainingStatus.PULLING_IMAGE,
                    config=config,
                    image=self.docker_image,
                    created_at=created_at,
                )
                self._preparing_jobs[job_id] = preparing_job
                try:
                    self._ensure_image()
                finally:
                    self._preparing_jobs.pop(job_id, None)

            # Prepare config file
            temp_config_path = self._dump_temp_config(config)

            # Prepare container labels with metadata
            labels = {
                DOCK_LABEL_KEY: DOCK_LABEL_VALUE,
                JOB_ID_LABEL_KEY: job_id,
                "llama-factory-dock.created_at": created_at.isoformat(),
                "llama-factory-dock.config": json.dumps(config) if config else "{}",
            }

            container_name = f"llama-factory-dock-{job_id}"

            self.logger.info(f"Starting container: {container_name}")
            self.logger.debug(f"Job ID: {job_id}")
            self.logger.debug(f"Volumes: dataset={self.dataset_dir}, recipe={self.recipe_dir}, output={self.output_dir}")

            # Start Docker container
            container = self.docker_client.containers.run(
                image=self.docker_image,
                platform=self.docker_container_platform,
                command=[
                    "llamafactory-cli", "train",
                    f"{self.docker_container_root}/temp_configs/{temp_config_path.name}",
                    *self._build_override_args(override_config),
                ],
                volumes={
                    **DOCKER_CONTAINER_VOLUME_MAP,
                    str(temp_config_path.parent): {"bind": str(DOCKER_CONTAINER_ROOT / "temp_configs"), "mode": "ro"},
                },
                shm_size=DOCKER_CONTAINER_SHM_SIZE,
                device_requests=DOCKER_CONTAINER_GPU_REQUEST,
                detach=True,
                name=container_name,
                labels=labels,
            )

            # Build TrainingJob from container
            job = self._container_to_job(container)
            self.logger.info(f"Training job started successfully: {job.job_id} (container: {job.container_id[:12]})")
            return job

        except Exception as e:
            # Return a failed job
            self.logger.error(f"Failed to start training job: {e}", exc_info=True)
            return TrainingJob(
                job_id="failed",
                container_id="",
                status=TrainingStatus.FAILED,
                error_message=str(e),
                created_at=datetime.now(),
            )


    def stop(self, job_or_container_id: str, timeout: int = 10, force: bool = False) -> TrainingJob:
        """Stop a training job (accepts job_id or container_id)"""
        self.logger.info(f"Stopping job: {job_or_container_id} (force={force})")
        container = self._resolve_container(job_or_container_id)
        wait_seconds = 1 if force else timeout
        container.stop(timeout=wait_seconds)
        container.reload()
        job = self._container_to_job(container)
        self.logger.info(f"Job stopped: {job.job_id}")
        return job

    def pause(self, job_or_container_id: str) -> TrainingJob:
        """Pause a training job (accepts job_id or container_id)"""
        self.logger.info(f"Pausing job: {job_or_container_id}")
        container = self._resolve_container(job_or_container_id)
        container.pause()
        container.reload()
        job = self._container_to_job(container)
        self.logger.info(f"Job paused: {job.job_id}")
        return job

    def resume(self, job_or_container_id: str) -> TrainingJob:
        """Resume a paused training job (accepts job_id or container_id)"""
        self.logger.info(f"Resuming job: {job_or_container_id}")
        container = self._resolve_container(job_or_container_id)
        container.unpause()
        container.reload()
        job = self._container_to_job(container)
        self.logger.info(f"Job resumed: {job.job_id}")
        return job

    def poll(self, job_or_container_id: str) -> TrainingJob:
        """Get job status (accepts job_id or container_id)"""
        if job_or_container_id in self._preparing_jobs:
            return self._preparing_jobs[job_or_container_id]
        container = self._resolve_container(job_or_container_id)
        container.reload()
        return self._container_to_job(container)

    def poll_logs(self, job_or_container_id: str, tail: int = 100) -> List[str]:
        """Get container logs (accepts job_id or container_id)"""
        try:
            container = self._resolve_container(job_or_container_id)
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            return logs.split('\n')
        except ValueError as e:
            return [str(e)]
        except Exception as e:
            return [f"Error getting logs: {str(e)}"]

    def list_jobs(self) -> List[TrainingJob]:
        """List all training jobs by querying Docker containers and preparing jobs"""
        containers = self.docker_client.containers.list(
            all=True,
            filters={"label": f"{DOCK_LABEL_KEY}={DOCK_LABEL_VALUE}"}
        )
        jobs = [self._container_to_job(c) for c in containers]
        for prep in self._preparing_jobs.values():
            if not any(j.job_id == prep.job_id for j in jobs):
                jobs.append(prep)
        return jobs

    def delete_job(self, job_or_container_id: str, force: bool = False) -> bool:
        """Delete a training job (accepts job_id or container_id)"""
        container = self._resolve_container(job_or_container_id)
        if force:
            container.remove(force=True)  # Kill if running, like docker rm -f
        else:
            container.stop(timeout=10)
            container.remove(force=False)
        return True

    def get_checkpoints(self, job_or_container_id: str) -> List[str]:
        """Get checkpoints for a job (accepts job_id or container_id)"""
        try:
            container = self._resolve_container(job_or_container_id)
            job = self._container_to_job(container)

            if not job.config:
                return []

            # Check output directory for checkpoints
            output_path = self.output_dir / job.config.get("output_dir", "")
            if not output_path.exists():
                return []

            checkpoints = []
            for checkpoint_dir in output_path.glob("checkpoint-*"):
                if checkpoint_dir.is_dir():
                    checkpoints.append(checkpoint_dir.name)

            return sorted(checkpoints)
        except Exception:
            return []

    def _resolve_container(self, job_or_container_id: str):
        """
        Resolve job_id or container_id to Docker container.
        Accepts either: job_id (8 chars from label) or container_id (12/64 chars).
        """
        # 1. Try as container ID first (Docker accepts short or full)
        try:
            container = self.docker_client.containers.get(job_or_container_id)
            self._verify_managed_container(container)
            return container
        except docker.errors.NotFound:
            pass

        # 2. Try as job_id via label - list managed containers, find by job_id
        containers = self.docker_client.containers.list(
            all=True,
            filters={"label": f"{DOCK_LABEL_KEY}={DOCK_LABEL_VALUE}"}
        )
        for c in containers:
            if (c.labels or {}).get(JOB_ID_LABEL_KEY) == job_or_container_id:
                return c

        raise ValueError(f"Job or container '{job_or_container_id}' not found")

    def _verify_managed_container(self, container) -> None:
        """Verify that a container is managed by llama-factory-dock"""
        labels = container.labels or {}
        if labels.get(DOCK_LABEL_KEY) != DOCK_LABEL_VALUE:
            raise ValueError(
                f"Container {container.short_id} is not managed by llama-factory-dock. "
                f"Only containers created by this tool can be managed."
            )

    def _container_to_job(self, container) -> TrainingJob:
        """Convert Docker container to TrainingJob"""
        # Get labels
        labels = container.labels or {}

        # Parse config from labels
        config_json = labels.get("llama-factory-dock.config", "{}")
        try:
            config = json.loads(config_json)
        except:
            config = {}

        # Parse timestamps
        created_at_str = labels.get("llama-factory-dock.created_at")
        try:
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
        except:
            created_at = datetime.now()

        # Map container status to our status
        container_status = container.status
        if container_status == "running":
            status = TrainingStatus.RUNNING
        elif container_status == "paused":
            status = TrainingStatus.PAUSED
        elif container_status == "exited":
            exit_code = container.attrs.get('State', {}).get('ExitCode', 1)
            status = TrainingStatus.COMPLETED if exit_code == 0 else TrainingStatus.FAILED
        elif container_status == "created":
            status = TrainingStatus.PENDING
        else:
            status = TrainingStatus.STOPPED

        # Get started_at and finished_at from container
        started_at_str = container.attrs.get('State', {}).get('StartedAt')
        finished_at_str = container.attrs.get('State', {}).get('FinishedAt')

        try:
            started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00')) if started_at_str else None
        except:
            started_at = None

        try:
            finished_at = datetime.fromisoformat(finished_at_str.replace('Z', '+00:00')) if finished_at_str else None
        except:
            finished_at = None

        # Build error message if failed
        error_message = None
        if status == TrainingStatus.FAILED:
            exit_code = container.attrs.get('State', {}).get('ExitCode', 'unknown')
            error_message = f"Container exited with code {exit_code}"

        job_id = labels.get(JOB_ID_LABEL_KEY, container.short_id)  # Prefer label, fallback to container_id

        # Get image from container (Config.Image or image.tags)
        image = None
        try:
            image = container.attrs.get("Config", {}).get("Image") or (
                container.image.tags[0] if container.image.tags else None
            )
        except Exception:
            pass

        return TrainingJob(
            job_id=job_id,
            container_id=container.short_id,
            status=status,
            config=config,
            image=image,
            created_at=created_at,
            started_at=started_at,
            completed_at=finished_at,
            error_message=error_message,
        )

    def _ensure_image(self):
        """Ensure Docker image exists, pull if needed"""
        try:
            self.docker_client.images.get(self.docker_image)
            self.logger.info(f"Docker image already exists: {self.docker_image}")
        except docker.errors.ImageNotFound:
            self.logger.info(f"Pulling Docker image: {self.docker_image} (this may take a while...)")
            self.logger.info(f"Note: Pulling x86_64 image on ARM64 Mac (Apple Silicon)")
            try:
                image = self.docker_client.images.pull(self.docker_image, platform="linux/amd64")
                self.logger.info(f"Image pulled successfully: {self.docker_image}")
                self.logger.debug(f"Image ID: {image.id}")
            except Exception as pull_error:
                raise RuntimeError(f"Failed to pull Docker image {self.docker_image}: {pull_error}") from pull_error

    @staticmethod
    def _build_override_args(overrides: Dict[str, Any]) -> List[str]:
        """Convert override dict to KEY=VALUE CLI args compatible with OmegaConf parsing.

        Boolean values are explicitly lowercased to match YAML semantics
        (e.g. True -> 'true'), since OmegaConf.from_cli() uses YAML-style parsing.
        """
        def _fmt(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            return str(v)

        return [f"{k}={_fmt(v)}" for k, v in overrides.items()]

    def get_train_help(self) -> str:
        """
        Get llamafactory-cli train --help output from the Docker image.

        Uses a two-layer cache keyed by image digest:
          1. In-memory: fastest, reset on process restart
          2. File cache (~/.cache/llama-factory-dock/): survives restarts, shared across processes

        If the image changes (e.g. after docker pull), the digest changes and both layers
        are bypassed automatically.
        """
        self._ensure_image()
        digest = self._get_image_digest()

        # Layer 1: in-memory
        if self._train_help_cache_key == digest and self._train_help_cache is not None:
            self.logger.debug("train_help: in-memory cache hit")
            return self._train_help_cache

        # Layer 2: file cache
        cached = self._read_train_help_file_cache(digest)
        if cached is not None:
            self.logger.debug("train_help: file cache hit")
            self._train_help_cache = cached
            self._train_help_cache_key = digest
            return self._train_help_cache

        # Cache miss: run container
        self.logger.info("train_help: fetching from container...")
        raw = self.docker_client.containers.run(
            image=self.docker_image,
            platform=self.docker_container_platform,
            command=["llamafactory-cli", "train", "--help"],
            remove=True,
            stdout=True,
            stderr=True,
        )
        result = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

        self._train_help_cache = result
        self._train_help_cache_key = digest
        self._write_train_help_file_cache(digest, result)
        self.logger.info("train_help: fetched and cached")
        return self._train_help_cache

    def prefetch_train_help(self) -> Optional[str]:
        """
        Prefetch train help at service startup.

        Non-throwing: logs a warning on failure instead of raising.
        Intended to be called during startup for cache warmup and image validation.
        """
        try:
            result = self.get_train_help()
            self.logger.info("train_help prefetch: success")
            return result
        except Exception as e:
            self.logger.warning(f"train_help prefetch failed (non-critical): {e}")
            return None

    def _get_image_digest(self) -> str:
        """Return local image digest (SHA256 ID). Assumes image already exists locally."""
        return self.docker_client.images.get(self.docker_image).id

    def _get_train_help_cache_file(self, digest: str) -> pathlib.Path:
        """Return path to file cache for the given image digest."""
        return TRAIN_HELP_CACHE_DIR / f"train-help-{digest.replace(':', '-')[:20]}.txt"

    def _read_train_help_file_cache(self, digest: str) -> Optional[str]:
        """Read train help from file cache. Returns None if not found or on error."""
        cache_file = self._get_train_help_cache_file(digest)
        try:
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.debug(f"train_help file cache read failed: {e}")
        return None

    def _write_train_help_file_cache(self, digest: str, content: str) -> None:
        """Write train help to file cache atomically. Silently skips on error."""
        cache_file = self._get_train_help_cache_file(digest)
        try:
            TRAIN_HELP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                dir=TRAIN_HELP_CACHE_DIR, delete=False, suffix=".tmp",
            ) as f:
                f.write(content)
                tmp_path = f.name
            os.replace(tmp_path, cache_file)
            self.logger.debug(f"train_help written to file cache: {cache_file}")
        except Exception as e:
            self.logger.debug(f"train_help file cache write failed (non-critical): {e}")

    def _dump_temp_config(self, config: Dict[str, Any]) -> pathlib.Path:
        """Dump temporary YAML config file"""
        config_dir = pathlib.Path(tempfile.gettempdir()) / "llama-factory-dock-configs"
        config_dir.mkdir(exist_ok=True)

        # Use timestamp to avoid conflicts
        config_filename = f"config-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.yaml"
        config_path = config_dir / config_filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path