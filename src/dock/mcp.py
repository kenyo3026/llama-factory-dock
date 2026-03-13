"""
FastMCP-based MCP interface for LlamaFactory Dock

Provides MCP tools for training control: start, stop, pause, resume.
"""

import argparse
import logging
import pathlib
from typing import Optional, Dict, Any, Union

from fastmcp import FastMCP

from .dock import LlamaFactoryDock, LlamaFactoryDryRunDock, DOCKER_CONTAINER_ROOT
from .utils.config_handler import parse_config_content, resolve_config
from .utils.logger import enable_rich_logger


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3028

# Resolve relative to package so dock-mcp works regardless of cwd
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "recipes" / "train_lora" / "qwen3_lora_sft.yaml")

MCP_NAME = "LlamaFactory Dock"
MCP_INSTRUCTION = """MCP server for LlamaFactory training orchestration.

**Background**: This MCP server provides tools to manage LlamaFactory training jobs via Docker containers.
LlamaFactory is a well-known unified framework for efficient LLM fine-tuning, supporting various models and training methods.

**Key Points**:
- All training configurations follow LlamaFactory's native format
- The server may be started with a base config (recipe); `override_config` in `start_training` merges on top of it
- Use the same parameter names and structure as LlamaFactory YAML config
- Common LlamaFactory parameters include: model_name_or_path, dataset, stage (pt/sft/rm/ppo/dpo/kto), learning_rate, num_train_epochs, per_device_train_batch_size, etc.

**Recommended Workflow**:
1. Call `get_server_config` to inspect the server's current base config (recipe), if any
2. Call `get_training_help` to discover all available parameters for the current LlamaFactory version
3. Construct an override_config dict based on what you want to change from the base config
4. Call `start_training` with override_config (or rely on the server base config entirely)

**Available Operations**: get train help, start, stop, pause, resume, and monitor training jobs. Each job runs in an isolated Docker container.
"""

MCP_START_TRAINING_INSTUCTION = """Start a new LlamaFactory training job in a Docker container.

**Config Resolution** (in priority order):
1. Server base config (--config at startup) is loaded as base, if provided.
2. override_config is merged on top of the base config (same keys win).
3. If neither base config nor override_config is provided, returns an error.

**override_config Format**: LlamaFactory config as a dictionary (JSON object).
Use the same keys as LlamaFactory's native YAML config.

**Configuration Tips**:
- Use standard LlamaFactory parameter names (model_name_or_path, dataset, stage, etc.)
- Common training stages: sft, pt, rm, ppo, dpo, kto
- Supports various finetuning methods: lora, full, freeze, etc.
- Call get_training_help() first to discover all valid parameter names.

**Server Base Config (Recipe)**:
This is the base training config loaded when the server started (via --config).
Your override_config will be merged ON TOP of this — only specify keys you want to change.
If this is None, your override_config must be a complete standalone config.
{base_training_config}

Args:
    override_config: Optional override dict merged on top of the server base config.
        If the server was not started with --config, this becomes the full config.

Returns:
    Job info dict with job_id (8 chars), container_id (12 chars), status, config, timestamps.
    Returns {{"error": "...", "status": "failed"}} on failure.
"""

def setup_mcp_server(
    logger: Optional[logging.Logger] = None,
    dryrun: bool = False,
    dryrun_duration: int = 300,
    prefetch_on_startup: bool = True,
    default_config: Optional[Union[str, pathlib.Path]] = None,
) -> FastMCP:
    """
    Create and configure FastMCP server for LlamaFactory Dock.

    Args:
        prefetch_on_startup: If True, prefetch train help at startup to warm the cache.
        default_config: Path to base config file (recipe). Used in
            start_training when override_config alone is provided. Validated at startup.

    Returns:
        Configured FastMCP server instance.
    """
    logger = logger or logging.getLogger(__name__)

    if default_config is not None and not pathlib.Path(default_config).exists():
        raise FileNotFoundError(f"Base config file not found: {default_config}")

    dock: LlamaFactoryDock = (
        LlamaFactoryDryRunDock(dryrun_training_duration=dryrun_duration, logger=logger)
        if dryrun else LlamaFactoryDock(logger=logger)
    )

    if prefetch_on_startup:
        logger.info("Prefetching train help at startup...")
        dock.prefetch_train_help()

    mcp = FastMCP(
        name=MCP_NAME,
        instructions=MCP_INSTRUCTION,
    )

    def start_training(override_config: Optional[Dict[str, Any]] = None) -> dict:
        try:
            base_config: Dict[str, Any] = {}
            if default_config is not None:
                base_config = parse_config_content(pathlib.Path(default_config).read_text())
                logger.info(f"start_training: loaded base config ({default_config})")

            override_config: Dict[str, Any] = override_config or {}

            if not base_config and not override_config:
                return {
                    "error": "Provide override_config, or start the server with --config",
                    "status": "failed",
                }

            if base_config:
                # Pass base config as YAML file; override_config becomes CLI key=value args
                # so llamafactory handles the merge at runtime (OmegaConf override semantics)
                logger.info("start_training: starting job with base config + override kwargs")
                job = dock.start(base_config, **override_config)
            else:
                # No base config — override_config is the full config
                resolved = resolve_config(override_config)
                logger.info("start_training: starting job with full override config")
                job = dock.start(resolved)

            logger.info(f"start_training: job started job_id={job.job_id} container_id={job.container_id}")
            return job.to_dict()
        except ValueError as e:
            logger.warning(f"start_training: invalid config: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"start_training: failed to start job: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    if default_config is not None:
        try:
            _config_yaml = pathlib.Path(default_config).read_text()
            _config_hint = f"Path: `{default_config}`\n```yaml\n{_config_yaml}```"
        except Exception:
            _config_hint = f"Path: `{default_config}` (unable to read)"
    else:
        _config_hint = "None"

    start_training.__doc__ = MCP_START_TRAINING_INSTUCTION.format(base_training_config=_config_hint)
    mcp.tool()(start_training)

    @mcp.tool()
    def resolve_training_cmd(override_config: Optional[Dict[str, Any]] = None) -> dict:
        """
        Resolve the full llamafactory-cli train command without launching a container.

        Returns the exact command that would be passed to the Docker container if
        start_training were called with the same override_config. Mirrors start_training's
        logic: base config goes into the YAML file, override_config becomes key=value CLI args.

        **Use Case**: Preview or validate the training command before committing to a real run.

        Args:
            override_config: Optional override dict (same as start_training).

        Returns:
            Dict with "command" (the full llamafactory-cli train ... string).
            Returns {"error": "...", "status": "failed"} on failure.
        """
        try:
            base_config: Dict[str, Any] = {}
            if default_config is not None:
                base_config = parse_config_content(pathlib.Path(default_config).read_text())

            override_config: Dict[str, Any] = override_config or {}

            if not base_config and not override_config:
                return {
                    "error": "Provide override_config, or start the server with --config",
                    "status": "failed",
                }

            config_placeholder = f"{DOCKER_CONTAINER_ROOT}/temp_configs/config.yaml"

            def _fmt(v: Any) -> str:
                return "true" if v is True else "false" if v is False else str(v)

            override_args = [f"{k}={_fmt(v)}" for k, v in override_config.items()]
            command = ["llamafactory-cli", "train", config_placeholder] + override_args

            logger.info(f"resolve_training_cmd: {' '.join(command)}")
            return {"command": " ".join(command)}
        except ValueError as e:
            logger.warning(f"resolve_training_cmd: invalid config: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"resolve_training_cmd: failed to resolve command: {e}", exc_info=True)
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
        [NOT IMPLEMENTED] Get training progress percentage for a job.

        **WARNING: This tool is not yet functional. Do NOT use this tool.**
        - progress_percentage always returns 0.0 (metrics parsing is not implemented)
        - Use `get_training_logs` instead to check actual training progress, epoch/step info, and errors.

        **Progress Calculation** (planned, not yet working):
        - Based on current_step / total_steps from training metrics
        - Returns 0.0 if metrics not available yet
        - Returns 100.0 when training completes

        Args:
            job_or_container_id: Either job ID (8 chars) or container ID (12 chars)

        Returns:
            Dict with job_id and progress_percentage (always 0.0 until implemented).
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

    @mcp.tool()
    def get_server_config() -> dict:
        """
        Get the server's current base config (recipe).

        **Use Case**: Call this before start_training to understand what base configuration
        the server was started with. This lets you decide which keys to pass in override_config
        and which are already covered by the base config.

        **When no base config is set**: Returns {"base_config": null, "config": null} to
        indicate start_training requires a full override_config.

        Returns:
            Dict with:
              - "base_config": path to the config file, or null if not set
              - "config": parsed config dict, or null if not set
        """
        if default_config is None:
            logger.debug("get_server_config: no base config set")
            return {"base_config": None, "config": None}
        try:
            parsed = parse_config_content(pathlib.Path(default_config).read_text())
            logger.debug(f"get_server_config: returning base config ({default_config})")
            return {"base_config": str(default_config), "config": parsed}
        except Exception as e:
            logger.error(f"get_server_config: failed to read base config: {e}", exc_info=True)
            return {"error": str(e), "status": "failed"}

    @mcp.tool()
    def get_training_help() -> dict:
        """
        Get llamafactory-cli train --help output from the current Docker image.

        **Use Case**: Discover all available training parameters, their types, defaults, and
        descriptions for the current LlamaFactory version. Call this before constructing a
        training config to ensure you are using valid parameter names.

        **Performance**: Result is cached by image digest (in-memory + file). The first call
        may take 30-60 seconds; subsequent calls return instantly from cache.

        Returns:
            Dict with "help_text" (str) containing the full train --help output.
            Returns {"error": "...", "status": "failed"} on failure.
        """
        logger.info("get_training_help: fetching train help")
        try:
            help_text = dock.get_train_help()
            logger.info("get_training_help: success")
            return {"help_text": help_text}
        except Exception as e:
            logger.error(f"get_training_help: failed: {e}", exc_info=True)
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
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Base config file (recipe) used when start_training is called",
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
    if args.config:
        logger.info(f"Base config: {args.config}")
    logger.info(f"Log directory: {logger_path.absolute()}")

    try:
        mcp = setup_mcp_server(
            logger=logger,
            dryrun=args.dryrun,
            dryrun_duration=args.dryrun_duration,
            default_config=args.config,
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
