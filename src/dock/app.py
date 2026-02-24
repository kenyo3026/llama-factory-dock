"""
FastAPI-based API interface for LlamaFactory Dock

Provides HTTP endpoints for training control, monitoring, and task management.
"""

import argparse
import json
import logging
import pathlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, File, Form, UploadFile
from pydantic import BaseModel, Field

from .dock import LlamaFactoryDock
from .utils.config_handler import merge_config, parse_config_content, resolve_config
from .utils.logger import enable_rich_logger

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3027

# --- Pydantic models ---

class StopRequest(BaseModel):
    """Request model for stop/delete with force option"""
    force: bool = Field(False, description="Force stop/remove (minimal wait)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")


class JobResponse(BaseModel):
    """Training job response (from TrainingJob.to_dict)"""
    job_id: str
    container_id: str
    status: str
    config: Optional[Dict] = None
    metrics: Optional[Dict] = None
    progress_percentage: float = 0.0
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class JobsListResponse(BaseModel):
    """List of training jobs"""
    jobs: List[JobResponse]
    total: int


class LogsResponse(BaseModel):
    """Logs response"""
    job_id: str
    logs: List[str]


class CheckpointsResponse(BaseModel):
    """Checkpoints list response"""
    job_id: str
    checkpoints: List[str]


def setup_app(
    dock_factory=None,
    logger: Optional[logging.Logger] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        dock_factory: Optional callable that returns LlamaFactoryDock instance.
                      Defaults to LlamaFactoryDock (real mode).
        logger: Optional logger instance. Uses module logger if not provided.

    Returns:
        Configured FastAPI application.
    """
    logger = logger or logging.getLogger(__name__)
    if dock_factory is None:
        dock_factory = LlamaFactoryDock

    app = FastAPI(
        title="LlamaFactory Dock API",
        description="API for Docker-based LlamaFactory training orchestration",
        version="0.1.0",
    )

    # Lazy init: create dock on first request (avoids Docker connection at import)
    _dock: Optional[LlamaFactoryDock] = None

    def get_dock() -> LlamaFactoryDock:
        nonlocal _dock
        if _dock is None:
            _dock = dock_factory() if callable(dock_factory) else dock_factory
        return _dock

    # --- Health ---
    @app.get("/", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(status="healthy", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check (alias)"""
        return HealthResponse(status="healthy", version="0.1.0")

    # --- Training control ---
    @app.post("/api/v1/training/start", response_model=JobResponse)
    async def start_training(
        config_file: Optional[UploadFile] = File(None, description="YAML/JSON config file (optional). Used as base when both provided."),
        config: Optional[str] = Form(
            None,
            description="Config as JSON string (optional). Overwrites same keys in config_file. Example: {\"learning_rate\": 0.001}. Leave empty if only using config_file.",
        ),
    ):
        """
        Start a training job.

        **config_file** (optional): YAML/JSON file upload.
        **config** (optional): Config as JSON string. Overwrites same keys in config_file.
        At least one required. If both: file as base, config overwrites same keys.
        """
        try:
            base_config: Dict[str, Any] = {}
            override_config: Dict[str, Any] = {}

            if config_file and config_file.filename:
                content = (await config_file.read()).decode("utf-8")
                base_config = parse_config_content(content)
                logger.info(f"start_training: config_file parsed ({config_file.filename})")

            if config and config.strip():
                try:
                    override_config = json.loads(config)
                except json.JSONDecodeError as e:
                    if base_config:
                        logger.warning(f"start_training: config field invalid JSON, ignoring (using config_file only): {e}")
                        override_config = {}
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"config must be valid JSON object, e.g. {{\"model_name_or_path\": \"llama2\", \"dataset\": \"alpaca\"}}. Error: {e}",
                        )
                else:
                    if not isinstance(override_config, dict):
                        raise HTTPException(status_code=400, detail="config must be a JSON object")

            if not base_config and not override_config:
                raise HTTPException(
                    status_code=400,
                    detail="Provide config and/or config_file (at least one required)",
                )

            config = merge_config(base_config, override_config) if base_config and override_config else (base_config or override_config)
            config = resolve_config(config)
            dock = get_dock()
            job = dock.start(config)
            logger.info(f"start_training: job started job_id={job.job_id} container_id={job.container_id}")
            if job.status == "failed":
                raise HTTPException(status_code=500, detail=job.error_message or "Failed to start")
            return JobResponse(**job.to_dict())

        except HTTPException:
            raise
        except ValueError as e:
            logger.warning(f"start_training: invalid config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"start_training: failed to start job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/training/{job_id}/stop", response_model=JobResponse)
    async def stop_training(job_id: str, force: bool = Query(False, description="Force stop")):
        """Stop a training job"""
        logger.info(f"stop_training: job_id={job_id} force={force}")
        try:
            job = get_dock().stop(job_id, force=force)
            logger.info(f"stop_training: job stopped job_id={job.job_id} status={job.status}")
            return JobResponse(**job.to_dict())
        except ValueError as e:
            logger.warning(f"stop_training: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"stop_training: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/training/{job_id}/pause", response_model=JobResponse)
    async def pause_training(job_id: str):
        """Pause a training job"""
        logger.info(f"pause_training: job_id={job_id}")
        try:
            job = get_dock().pause(job_id)
            logger.info(f"pause_training: job paused job_id={job.job_id}")
            return JobResponse(**job.to_dict())
        except ValueError as e:
            logger.warning(f"pause_training: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"pause_training: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/training/{job_id}/resume", response_model=JobResponse)
    async def resume_training(job_id: str):
        """Resume a paused training job"""
        logger.info(f"resume_training: job_id={job_id}")
        try:
            job = get_dock().resume(job_id)
            logger.info(f"resume_training: job resumed job_id={job.job_id}")
            return JobResponse(**job.to_dict())
        except ValueError as e:
            logger.warning(f"resume_training: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"resume_training: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # --- Monitoring ---
    @app.get("/api/v1/training/{job_id}/status", response_model=JobResponse)
    async def get_status(job_id: str):
        """Get job status (poll)"""
        logger.debug(f"get_status: job_id={job_id}")
        try:
            job = get_dock().poll(job_id)
            return JobResponse(**job.to_dict())
        except ValueError as e:
            logger.warning(f"get_status: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"get_status: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/logs", response_model=LogsResponse)
    async def get_logs(
        job_id: str,
        tail: int = Query(100, ge=1, le=10000, description="Number of log lines"),
    ):
        """Get job logs"""
        logger.debug(f"get_logs: job_id={job_id} tail={tail}")
        try:
            logs = get_dock().poll_logs(job_id, tail=tail)
            return LogsResponse(job_id=job_id, logs=logs)
        except ValueError as e:
            logger.warning(f"get_logs: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"get_logs: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/progress")
    async def get_progress(job_id: str):
        """Get training progress (percentage)"""
        logger.debug(f"get_progress: job_id={job_id}")
        try:
            job = get_dock().poll(job_id)
            return {"job_id": job_id, "progress_percentage": job.get_progress_percentage()}
        except ValueError as e:
            logger.warning(f"get_progress: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"get_progress: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # --- Task management ---
    @app.get("/api/v1/training/jobs", response_model=JobsListResponse)
    async def list_jobs():
        """List all training jobs"""
        logger.info("list_jobs: listing all jobs")
        try:
            jobs = get_dock().list_jobs()
            logger.info(f"list_jobs: found {len(jobs)} job(s)")
            job_responses = [JobResponse(**j.to_dict()) for j in jobs]
            return JobsListResponse(jobs=job_responses, total=len(job_responses))
        except Exception as e:
            logger.error(f"list_jobs: failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/training/{job_id}")
    async def delete_job(
        job_id: str,
        force: bool = Query(False, description="Force remove (kill if running)"),
    ):
        """Delete a training job"""
        logger.info(f"delete_job: job_id={job_id} force={force}")
        try:
            get_dock().delete_job(job_id, force=force)
            logger.info(f"delete_job: job deleted job_id={job_id}")
            return {"job_id": job_id, "deleted": True}
        except ValueError as e:
            logger.warning(f"delete_job: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"delete_job: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/checkpoints", response_model=CheckpointsResponse)
    async def get_checkpoints(job_id: str):
        """List checkpoints for a job"""
        logger.debug(f"get_checkpoints: job_id={job_id}")
        try:
            checkpoints = get_dock().get_checkpoints(job_id)
            return CheckpointsResponse(job_id=job_id, checkpoints=checkpoints)
        except ValueError as e:
            logger.warning(f"get_checkpoints: job not found job_id={job_id}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"get_checkpoints: failed job_id={job_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = setup_app()


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    dock_factory=None,
    logger: Optional[logging.Logger] = None,
    reload: bool = False,
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        dock_factory: Optional callable returning LlamaFactoryDock (real mode by default)
        logger: Optional logger instance
        reload: Enable auto-reload for development
    """
    import uvicorn

    api_app = setup_app(dock_factory=dock_factory, logger=logger)
    uvicorn.run(api_app, host=host, port=port, reload=reload)


def main() -> int:
    """Main entry point for dock-api command"""
    parser = argparse.ArgumentParser(description="Run LlamaFactory Dock API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory (logs go to checkpoint/logs)")
    parser.add_argument("--dryrun", default=None, help="")

    args = parser.parse_args()

    # Initialize logger (use absolute path so logs work regardless of cwd)
    if args.checkpoint:
        logger_path = pathlib.Path(args.checkpoint).resolve() / "logs"
    else:
        logger_path = pathlib.Path.cwd() / "logs"

    logger = enable_rich_logger(
        directory=logger_path,
        name="llama-factory-dock-api",
    )

    logger.info("Starting LlamaFactory Dock API server")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Log directory: {logger_path.absolute()}")

    try:
        run_server(
            host=args.host,
            port=args.port,
            logger=logger,
            reload=args.reload,
        )
    except Exception as e:
        logger.error(f"API server error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
