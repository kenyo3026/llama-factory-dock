"""
FastMCP-based MCP interface for LlamaFactory Dock

Provides MCP tools for training control: start, stop, pause, resume.
"""

import argparse
import logging
import pathlib
from typing import Optional, Dict, Any

from fastmcp import FastMCP

from .dock import LlamaFactoryDock, LlamaFactoryDryRunDock
from .utils.config_handler import resolve_config
from .utils.logger import enable_rich_logger


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3028

MCP_NAME = "LlamaFactory Dock"
MCP_INSTRUCTION = """MCP server for LlamaFactory training orchestration.

**Background**: This MCP server provides tools to manage LlamaFactory training jobs via Docker containers.
LlamaFactory is a well-known unified framework for efficient LLM fine-tuning, supporting various models and training methods.

**Key Points**:
- All training configurations follow LlamaFactory's native format
- The `config` parameter in `start_training` accepts a dictionary (JSON object) with LlamaFactory config
- Use the same parameter names and structure as LlamaFactory YAML config
- Common LlamaFactory parameters include: model_name_or_path, dataset, stage (pt/sft/rm/ppo/dpo/kto), learning_rate, num_train_epochs, per_device_train_batch_size, etc.

**Available Operations**: start, stop, pause, and resume training jobs. Each job runs in an isolated Docker container.
"""

def setup_mcp_server(
    logger: Optional[logging.Logger] = None,
    dryrun: bool = False,
    dryrun_duration: int = 300,
) -> FastMCP:
    """
    Create and configure FastMCP server for LlamaFactory Dock.

    Returns:
        Configured FastMCP server instance.
    """
    logger = logger or logging.getLogger(__name__)

    dock: LlamaFactoryDock = (
        LlamaFactoryDryRunDock(dryrun_training_duration=dryrun_duration, logger=logger)
        if dryrun else LlamaFactoryDock(logger=logger)
    )

    mcp = FastMCP(
        name=MCP_NAME,
        instructions=MCP_INSTRUCTION,
    )

    @mcp.tool()
    def start_training(config: Dict[str, Any]) -> dict:
        """
        Start a new LlamaFactory training job in a Docker container.

        **Config Format**: LlamaFactory config as a dictionary (JSON object).
        Use the same keys as LlamaFactory's native YAML config.

        **Configuration Tips**:
        - Use standard LlamaFactory parameter names (model_name_or_path, dataset, stage, etc.)
        - Refer to LlamaFactory documentation or examples for parameter details
        - Common training stages: sft, pt, rm, ppo, dpo, kto
        - Supports various finetuning methods: lora, full, freeze, etc.

        Args:
            config: LlamaFactory config as dictionary. Same structure as LlamaFactory YAML config.

        Returns:
            Job info dict with job_id (8 chars), container_id (12 chars), status, config, timestamps.
            Returns {"error": "...", "status": "failed"} on failure.
        """
        try:
            resolved = resolve_config(config)
            logger.info("start_training: starting job with config=dict")
            job = dock.start(resolved)
            logger.info(f"start_training: job started job_id={job.job_id} container_id={job.container_id}")
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"start_training: invalid config: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"start_training: failed to start job: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def stop_training(job_or_container_id: str, force: bool = False) -> dict:
        """
        Stop a running or paused training job gracefully or forcefully.

        **Behavior**:
        - Normal stop (force=False): Graceful shutdown with 10s timeout
        - Force stop (force=True): Immediate termination with 1s timeout

        **When to use force**: Job is unresponsive, stuck, or immediate termination needed.

        **ID Formats**:
        - Job ID: 8-character alphanumeric (returned by start_training)
        - Container ID: 12-character hex (from Docker, also in start_training response)

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)
            force: If True, force immediate stop; if False, graceful stop with timeout

        Returns:
            Job info dict with updated status after stop operation.
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.info(f"stop_training: job_or_container_id={job_or_container_id} force={force}")
        try:
            job = dock.stop(job_or_container_id, force=force)
            logger.info(f"stop_training: job stopped job_id={job.job_id} status={job.status}")
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"stop_training: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"stop_training: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def pause_training(job_or_container_id: str) -> dict:
        """
        Pause a running training job (sends SIGSTOP to container process).

        **Use Case**: Temporarily halt training to free up GPU/CPU resources without losing progress.
        The job can be resumed later with resume_training.

        **Note**:
        - Only works on jobs in "running" status
        - Container remains alive but process is frozen
        - No checkpoints are saved during pause

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Job info dict with status changed to "paused".
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.info(f"pause_training: job_or_container_id={job_or_container_id}")
        try:
            job = dock.pause(job_or_container_id)
            logger.info(f"pause_training: job paused job_id={job.job_id}")
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"pause_training: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"pause_training: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def resume_training(job_or_container_id: str) -> dict:
        """
        Resume a paused training job (sends SIGCONT to container process).

        **Use Case**: Continue training after pause_training was called.
        Training resumes from where it was frozen.

        **Note**:
        - Only works on jobs in "paused" status
        - Training continues without data loss
        - GPU/CPU resources are reacquired

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Job info dict with status changed back to "running".
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.info(f"resume_training: job_or_container_id={job_or_container_id}")
        try:
            job = dock.resume(job_or_container_id)
            logger.info(f"resume_training: job resumed job_id={job.job_id}")
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"resume_training: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"resume_training: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def get_training_status(job_or_container_id: str) -> dict:
        """
        Get current status and details of a training job.

        **Use Case**: Poll job status to monitor training progress, check if job completed/failed.

        **Returned Info**:
        - status: Current state (pending, running, paused, exited, failed)
        - progress_percentage: Training progress (0-100)
        - metrics: Training metrics if available (loss, accuracy, etc.)
        - timestamps: created_at, started_at, completed_at
        - error_message: If job failed

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Full job info dict with all available details.
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.debug(f"get_training_status: job_or_container_id={job_or_container_id}")
        try:
            job = dock.poll(job_or_container_id)
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"get_training_status: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"get_training_status: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def get_training_logs(job_or_container_id: str, tail: int = 100) -> dict:
        """
        Retrieve training logs from a job's container.

        **Use Case**: Monitor training output, debug issues, check training metrics in real-time.

        **Notes**:
        - Returns most recent N lines (default: 100, max: 10000)
        - Logs are from container stdout/stderr
        - Works for running, paused, and stopped jobs

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)
            tail: Number of recent log lines to retrieve (1-10000, default: 100)

        Returns:
            Dict with job_id and logs (list of strings).
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.debug(f"get_training_logs: job_or_container_id={job_or_container_id} tail={tail}")
        try:
            if tail < 1 or tail > 10000:
                logger.warning(f"get_training_logs: invalid tail={tail}")
                return {"error": "tail must be between 1 and 10000", "status": "invalid_param"}
            logs = dock.poll_logs(job_or_container_id, tail=tail)
            return {"job_id": job_or_container_id, "logs": logs}
        except ValueError as e:
            logger.warning(f"get_training_logs: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"get_training_logs: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def get_training_progress(job_or_container_id: str) -> dict:
        """
        Get training progress percentage for a job.

        **Use Case**: Quick check of how far training has progressed.

        **Progress Calculation**:
        - Based on current_step / total_steps from training metrics
        - Returns 0.0 if metrics not available yet
        - Returns 100.0 when training completes

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Dict with job_id and progress_percentage (0.0-100.0).
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.debug(f"get_training_progress: job_or_container_id={job_or_container_id}")
        try:
            job = dock.poll(job_or_container_id)
            return {
                "job_id": job_or_container_id,
                "progress_percentage": job.get_progress_percentage()
            }
        except ValueError as e:
            logger.warning(f"get_training_progress: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"get_training_progress: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def list_training_jobs() -> dict:
        """
        List all training jobs managed by this Dock instance.

        **Use Case**: Get overview of all jobs, find job IDs, check overall training status.

        **Notes**:
        - Only shows jobs with the llama-factory-dock label
        - Includes jobs in all states (running, paused, stopped, failed)
        - Jobs are identified by Docker labels

        Returns:
            Dict with "jobs" (list of job info dicts) and "total" (count).
            Each job dict contains: job_id, container_id, status, progress, timestamps, etc.
        """
        logger.info("list_training_jobs: listing all jobs")
        try:
            jobs = dock.list_jobs()
            logger.info(f"list_training_jobs: found {len(jobs)} job(s)")
            return {
                "jobs": [j.to_dict() for j in jobs],
                "total": len(jobs)
            }
        except Exception as e:
            logger.error(f"list_training_jobs: failed: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def delete_training_job(job_or_container_id: str, force: bool = False) -> dict:
        """
        Delete a training job and remove its container.

        **Behavior**:
        - Normal delete (force=False): Stops job gracefully (10s timeout), then removes container
        - Force delete (force=True): Kills job immediately (1s timeout), force-removes container

        **Warning**: This permanently removes the container. Checkpoints in mounted volumes are preserved.

        **When to use force**:
        - Job won't stop normally
        - Need immediate cleanup
        - Container is in bad state

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)
            force: If True, force kill and remove; if False, graceful stop then remove

        Returns:
            Dict with job_id and deleted=True on success.
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.info(f"delete_training_job: job_or_container_id={job_or_container_id} force={force}")
        try:
            dock.delete_job(job_or_container_id, force=force)
            logger.info(f"delete_training_job: job deleted job_or_container_id={job_or_container_id}")
            return {"job_id": job_or_container_id, "deleted": True}
        except ValueError as e:
            logger.warning(f"delete_training_job: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"delete_training_job: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def list_training_checkpoints(job_or_container_id: str) -> dict:
        """
        List saved checkpoints for a training job.

        **Use Case**: Find available checkpoints for model evaluation, recovery, or deployment.

        **Notes**:
        - Checkpoints are in the job's output_dir (from config)
        - Typically named like: checkpoint-100, checkpoint-500, etc.
        - Only works if output_dir is properly configured

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Dict with job_id and checkpoints (list of checkpoint directory names).
            Returns {"error": "...", "status": "not_found"} if job doesn't exist.
        """
        logger.debug(f"list_training_checkpoints: job_or_container_id={job_or_container_id}")
        try:
            checkpoints = dock.get_checkpoints(job_or_container_id)
            return {"job_id": job_or_container_id, "checkpoints": checkpoints}
        except ValueError as e:
            logger.warning(f"list_training_checkpoints: job not found job_or_container_id={job_or_container_id}: {e}")
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
            logger.error(f"list_training_checkpoints: failed job_or_container_id={job_or_container_id}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    return mcp


def main() -> int:
    """Main entry point for dock-mcp command"""
    parser = argparse.ArgumentParser(
        description="Run LlamaFactory Dock MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dock-mcp --transport stdio
  dock-mcp --transport streamable-http --port 8002
""",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host for streamable-http")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for streamable-http")
    parser.add_argument("--path", default="/mcp", help="Path for streamable-http")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use LlamaFactoryDryRunDock (simulated training, no GPU)",
    )
    parser.add_argument(
        "--dryrun-duration",
        type=int,
        default=300,
        help="Dryrun simulated training duration in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Initialize logger (use absolute path so logs work regardless of cwd)
    if args.checkpoint:
        logger_path = pathlib.Path(args.checkpoint).resolve() / 'logs'
    else:
        logger_path = pathlib.Path.cwd() / 'logs'

    logger = enable_rich_logger(
        directory=logger_path,
        name='llama-factory-dock-mcp',
    )

    logger.info("Starting LlamaFactory Dock MCP server")
    logger.info(f"Mode: {'dryrun' if args.dryrun else 'normal'}")
    if args.dryrun:
        logger.info(f"Dryrun duration: {args.dryrun_duration}s")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Log directory: {logger_path.absolute()}")

    try:
        mcp = setup_mcp_server(
            logger=logger,
            dryrun=args.dryrun,
            dryrun_duration=args.dryrun_duration,
        )
        if args.transport == "stdio":
            logger.info("Running MCP server with stdio transport")
            mcp.run(transport="stdio", show_banner=False)
        else:
            logger.info(f"Running MCP server with streamable-http transport: {args.host}:{args.port}{args.path}")
            # User-friendly console output
            logger.info(f"LlamaFactory Dock MCP server listening on http://{args.host}:{args.port}{args.path}")
            mcp.run(
                transport="streamable-http",
                host=args.host,
                port=args.port,
                path=args.path,
            )
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
