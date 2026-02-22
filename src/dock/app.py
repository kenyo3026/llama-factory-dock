"""
FastAPI-based API interface for LlamaFactory Dock

Provides HTTP endpoints for training control, monitoring, and task management.
"""

import argparse
from typing import Optional, List, Union, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .dock import LlamaFactoryDock

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3027

# --- Pydantic models ---

class StartRequest(BaseModel):
    """Request model for starting a training job"""
    config_path: Optional[str] = Field(
        None,
        description="Path to LlamaFactory config file (YAML/JSON)",
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="LlamaFactory config as dict (alternative to config_path)",
    )


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


def setup_app(dock_factory=None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        dock_factory: Optional callable that returns LlamaFactoryDock instance.
                      Defaults to LlamaFactoryDock (real mode).

    Returns:
        Configured FastAPI application.
    """
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
    async def start_training(request: StartRequest):
        """
        Start a training job.

        Provide either config_path (file path) or config (dict).
        """
        if request.config_path and request.config:
            raise HTTPException(
                status_code=400,
                detail="Provide either config_path or config, not both",
            )
        if not request.config_path and not request.config:
            raise HTTPException(
                status_code=400,
                detail="Provide config_path or config",
            )

        config: Union[str, Dict] = request.config_path or request.config
        dock = get_dock()
        job = dock.start(config)

        if job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error_message or "Failed to start")

        return JobResponse(**job.to_dict())

    @app.post("/api/v1/training/{job_id}/stop", response_model=JobResponse)
    async def stop_training(job_id: str, force: bool = Query(False, description="Force stop")):
        """Stop a training job"""
        try:
            job = get_dock().stop(job_id, force=force)
            return JobResponse(**job.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/training/{job_id}/pause", response_model=JobResponse)
    async def pause_training(job_id: str):
        """Pause a training job"""
        try:
            job = get_dock().pause(job_id)
            return JobResponse(**job.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/training/{job_id}/resume", response_model=JobResponse)
    async def resume_training(job_id: str):
        """Resume a paused training job"""
        try:
            job = get_dock().resume(job_id)
            return JobResponse(**job.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- Monitoring ---
    @app.get("/api/v1/training/{job_id}/status", response_model=JobResponse)
    async def get_status(job_id: str):
        """Get job status (poll)"""
        try:
            job = get_dock().poll(job_id)
            return JobResponse(**job.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/logs", response_model=LogsResponse)
    async def get_logs(
        job_id: str,
        tail: int = Query(100, ge=1, le=10000, description="Number of log lines"),
    ):
        """Get job logs"""
        try:
            logs = get_dock().poll_logs(job_id, tail=tail)
            return LogsResponse(job_id=job_id, logs=logs)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/progress")
    async def get_progress(job_id: str):
        """Get training progress (percentage)"""
        try:
            job = get_dock().poll(job_id)
            return {"job_id": job_id, "progress_percentage": job.get_progress_percentage()}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- Task management ---
    @app.get("/api/v1/training/jobs", response_model=JobsListResponse)
    async def list_jobs():
        """List all training jobs"""
        try:
            jobs = get_dock().list_jobs()
            job_responses = [JobResponse(**j.to_dict()) for j in jobs]
            return JobsListResponse(jobs=job_responses, total=len(job_responses))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/training/{job_id}")
    async def delete_job(
        job_id: str,
        force: bool = Query(False, description="Force remove (kill if running)"),
    ):
        """Delete a training job"""
        try:
            get_dock().delete_job(job_id, force=force)
            return {"job_id": job_id, "deleted": True}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/training/{job_id}/checkpoints", response_model=CheckpointsResponse)
    async def get_checkpoints(job_id: str):
        """List checkpoints for a job"""
        try:
            checkpoints = get_dock().get_checkpoints(job_id)
            return CheckpointsResponse(job_id=job_id, checkpoints=checkpoints)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Default app instance
app = setup_app()


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    dock_factory=None,
    reload: bool = False,
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        dock_factory: Optional callable returning LlamaFactoryDock (real mode by default)
        reload: Enable auto-reload for development
    """
    import uvicorn

    api_app = setup_app(dock_factory=dock_factory)
    uvicorn.run(api_app, host=host, port=port, reload=reload)


def main():
    """Main entry point for dock-api command"""
    parser = argparse.ArgumentParser(description="Run LlamaFactory Dock API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
