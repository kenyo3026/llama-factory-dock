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


DOCKER_IMAGE: str = "hiyouga/llamafactory:latest"
DOCKER_CONTAINER_PLATFORM: str = "linux/amd64"  # Force x86_64 on ARM64 (slower but works)
DOCKER_CONTAINER_ROOT: str = "/app/data"

PROJECT_ROOT = pathlib.Path(os.getcwd())
DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
OUTPUT_DIR: pathlib.Path = PROJECT_ROOT / "output"

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
        self.data_dir = pathlib.Path(DATA_DIR)
        self.output_dir = pathlib.Path(OUTPUT_DIR)

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory jobs during preparation (e.g. PULLING_IMAGE) before container exists
        self._preparing_jobs: Dict[str, TrainingJob] = {}

        self.logger.info(f"LlamaFactoryDock initialized")
        self.logger.debug(f"Docker image: {self.docker_image}")
        self.logger.debug(f"Platform: {self.docker_container_platform}")
        self.logger.debug(f"Data directory: {self.data_dir}")
        self.logger.debug(f"Output directory: {self.output_dir}")

    def start(
        self,
        config: Union[None, str, pathlib.Path, Dict[str, Any]],
        auto_pull: bool = True,
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
            self.logger.debug(f"Volumes: data={self.data_dir}, output={self.output_dir}")

            # Start Docker container
            container = self.docker_client.containers.run(
                image=self.docker_image,
                platform=self.docker_container_platform,
                command=[
                    "llamafactory-cli", "train",
                    f"{self.docker_container_root}/temp_configs/{temp_config_path.name}"
                ],
                volumes={
                    str(self.data_dir): {"bind": f"{self.docker_container_root}/datasets", "mode": "ro"},
                    str(self.output_dir): {"bind": f"{self.docker_container_root}/outputs", "mode": "rw"},
                    str(temp_config_path.parent): {"bind": f"{self.docker_container_root}/temp_configs", "mode": "ro"},
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=[str(i) for i in config.get('gpu_ids', [])] if config.get('gpu_ids') else None,
                        count=-1 if not config.get('gpu_ids') else len(config.get('gpu_ids', [])),
                        capabilities=[['gpu']]
                    )
                ],
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