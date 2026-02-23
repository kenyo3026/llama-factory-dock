import json
import uuid
import pathlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

from .utils.loader import ConfigLoader
from .instance import (
    LlamaFactoryDock,
    TrainingJob,
    TrainingStatus,
    DOCK_LABEL_KEY,
    DOCK_LABEL_VALUE,
    JOB_ID_LABEL_KEY,
)

DOCKER_IMAGE = "python:3.11-slim"


class LlamaFactoryDryRunDock(LlamaFactoryDock):
    """
    Dryrun version of LlamaFactoryDock for demo/testing.

    - Uses lightweight Docker image (python:3.11-slim, ARM64 compatible)
    - Starts real containers (pause/resume/stop work normally)
    - Simulates training with fake logs (no actual training)
    - Doesn't consume GPU resources
    """

    def __init__(self, dryrun_training_duration: int = 3600, logger: Optional[logging.Logger] = None):
        """
        Initialize dryrun dock

        Args:
            dryrun_training_duration: Duration of simulated training in seconds (default: 3600 = 1 hour)
            logger: Optional logger instance
        """
        super().__init__(logger=logger)
        self.dryrun_training_duration = dryrun_training_duration
        self.docker_image = DOCKER_IMAGE
        self.logger.info(f"LlamaFactoryDryRunDock initialized (dryrun mode)")
        self.logger.debug(f"Dryrun duration: {dryrun_training_duration}s")

    def start(
        self,
        config: Union[None, str, pathlib.Path, Dict[str, Any]],
        auto_pull: bool = True,
    ) -> TrainingJob:
        """Start a dryrun training job (container with simulated logs)"""
        self.logger.info(f"[DRYRUN] Starting training job with config type: {type(config).__name__}")

        if isinstance(config, str):
            config = ConfigLoader.load(config)

        try:
            if auto_pull:
                self._ensure_image()

            temp_config_path = self._dump_temp_config(config)
            dryrun_command = self._build_dryrun_training_command()

            created_at = datetime.now()
            job_id = uuid.uuid4().hex[:8]
            labels = {
                DOCK_LABEL_KEY: DOCK_LABEL_VALUE,
                JOB_ID_LABEL_KEY: job_id,
                "llama-factory-dock.created_at": created_at.isoformat(),
                "llama-factory-dock.config": json.dumps(config) if config else "{}",
                "llama-factory-dock.mode": "dryrun",
            }

            container_name = f"llama-factory-dock-dryrun-{job_id}"
            container = self.docker_client.containers.run(
                image=self.docker_image,
                command=["sh", "-c", dryrun_command],
                volumes={
                    str(self.data_dir): {"bind": "/app/data", "mode": "ro"},
                    str(self.output_dir): {"bind": "/app/output", "mode": "rw"},
                    str(temp_config_path.parent): {"bind": "/app/configs", "mode": "ro"},
                },
                detach=True,
                name=container_name,
                labels=labels,
            )

            job = self._container_to_job(container)
            self.logger.info(f"[DRYRUN] Training job started: {job.job_id} (container: {job.container_id[:12]})")
            return job

        except Exception as e:
            self.logger.error(f"[DRYRUN] Failed to start training job: {e}", exc_info=True)
            return TrainingJob(
                job_id="failed",
                container_id="",
                status=TrainingStatus.FAILED,
                error_message=str(e),
                created_at=datetime.now(),
            )

    def _build_dryrun_training_command(self) -> str:
        """Build dryrun training command that simulates realistic training logs"""
        total_steps = self.dryrun_training_duration

        return f"""
echo '[DRYRUN MODE] Starting simulated training'
echo 'Duration: {total_steps} seconds'
echo 'Config: /app/configs/'
echo '================================'
echo ''

for i in $(seq 1 {total_steps}); do
    epoch=$((($i - 1) / 100 + 1))
    step=$(($i % 100))
    if [ $step -eq 0 ]; then step=100; fi

    # Simulated loss (decreasing over time)
    loss=$(awk "BEGIN {{printf \\"%.4f\\", 2.5 - $i * 0.0005}}")

    # Simulated accuracy (increasing over time)
    acc=$(awk "BEGIN {{printf \\"%.2f\\", 20 + $i * 0.01}}")

    echo "Epoch $epoch | Step $step/100 | Loss: $loss | Accuracy: $acc%"

    # Occasional checkpoint messages
    if [ $(($i % 500)) -eq 0 ]; then
        echo ">>> Saving checkpoint at step $i"
    fi

    sleep 1
done

echo ''
echo '================================'
echo '[DRYRUN MODE] Training completed!'
"""
