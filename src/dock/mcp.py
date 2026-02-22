"""
FastMCP-based MCP interface for LlamaFactory Dock

Provides MCP tools for training control: start, stop, pause, resume.
"""

import argparse
from typing import Optional, Dict, Any, Union

from fastmcp import FastMCP

from .dock import LlamaFactoryDock


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3028

MCP_NAME = "LlamaFactory Dock"
MCP_INSTRUCTION = """MCP server for LlamaFactory training orchestration.

**Background**: This MCP server provides tools to manage LlamaFactory training jobs via Docker containers.
LlamaFactory is a well-known unified framework for efficient LLM fine-tuning, supporting various models and training methods.

**Key Points**:
- All training configurations follow LlamaFactory's native YAML/JSON format
- The `config` parameter in `start_training` accepts standard LlamaFactory configuration dictionaries
- If you're familiar with LlamaFactory, use the same parameter names and structure you would use with `llamafactory-cli train`
- Common LlamaFactory parameters include: model_name_or_path, dataset, stage (pt/sft/rm/ppo/dpo/kto), learning_rate, num_train_epochs, per_device_train_batch_size, etc.

**Available Operations**: start, stop, pause, and resume training jobs. Each job runs in an isolated Docker container.
"""

# Lazy init: create dock on first tool call
_dock: Optional[LlamaFactoryDock] = None


def get_dock() -> LlamaFactoryDock:
    """Get dock instance (lazy init)"""
    global _dock
    if _dock is None:
        _dock = LlamaFactoryDock()
    return _dock


def setup_mcp_server() -> FastMCP:
    """
    Create and configure FastMCP server for LlamaFactory Dock.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP(
        name=MCP_NAME,
        instructions=MCP_INSTRUCTION,
    )

    @mcp.tool()
    def start_training(
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Start a new LlamaFactory training job in a Docker container.

        **Usage**: Provide EITHER config_path (path to file) OR config (dict), not both.

        **Config Format**: Must follow LlamaFactory's standard configuration format.
        - For file: Accepts YAML (.yaml/.yml) or JSON (.json) files
        - For dict: Use the same keys as LlamaFactory's native YAML config

        **Configuration Tips**:
        - Use standard LlamaFactory parameter names (model_name_or_path, dataset, stage, etc.)
        - Refer to LlamaFactory documentation or examples for parameter details
        - Common training stages: sft, pt, rm, ppo, dpo, kto
        - Supports various finetuning methods: lora, full, freeze, etc.

        Args:
            config_path: Path to LlamaFactory config file (YAML/JSON). Mutually exclusive with `config`.
            config: LlamaFactory config as dictionary. Mutually exclusive with `config_path`.

        Returns:
            Job info dict with job_id (8 chars), container_id (12 chars), status, config, timestamps.
            Returns {"error": "...", "status": "failed"} on failure.
        """
        if config_path and config:
            return {"error": "Provide either config_path or config, not both"}
        if not config_path and not config:
            return {"error": "Provide config_path or config"}

        cfg: Union[str, Dict] = config_path or config
        try:
            job = get_dock().start(cfg)
            return job.to_dict()
        except Exception as e:
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
        try:
            job = get_dock().stop(job_or_container_id, force=force)
            return job.to_dict()
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            job = get_dock().pause(job_or_container_id)
            return job.to_dict()
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            job = get_dock().resume(job_or_container_id)
            return job.to_dict()
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            job = get_dock().poll(job_or_container_id)
            return job.to_dict()
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            if tail < 1 or tail > 10000:
                return {"error": "tail must be between 1 and 10000", "status": "invalid_param"}
            logs = get_dock().poll_logs(job_or_container_id, tail=tail)
            return {"job_id": job_or_container_id, "logs": logs}
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            job = get_dock().poll(job_or_container_id)
            return {
                "job_id": job_or_container_id,
                "progress_percentage": job.get_progress_percentage()
            }
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            jobs = get_dock().list_jobs()
            return {
                "jobs": [j.to_dict() for j in jobs],
                "total": len(jobs)
            }
        except Exception as e:
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
        try:
            get_dock().delete_job(job_or_container_id, force=force)
            return {"job_id": job_or_container_id, "deleted": True}
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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
        try:
            checkpoints = get_dock().get_checkpoints(job_or_container_id)
            return {"job_id": job_or_container_id, "checkpoints": checkpoints}
        except ValueError as e:
            return {"error": str(e), "status": "not_found"}
        except Exception as e:
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

    args = parser.parse_args()

    try:
        mcp = setup_mcp_server()
        if args.transport == "stdio":
            mcp.run(transport="stdio", show_banner=False)
        else:
            print(f"LlamaFactory Dock MCP server", flush=True)
            print(f"  Host: {args.host}  Port: {args.port}  Path: {args.path}", flush=True)
            mcp.run(
                transport="streamable-http",
                host=args.host,
                port=args.port,
                path=args.path,
            )
    except Exception as e:
        import sys
        import traceback
        print(f"Error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
