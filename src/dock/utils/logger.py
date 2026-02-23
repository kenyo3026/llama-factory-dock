import logging
import sys
import os
import pathlib
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from rich.logging import RichHandler
from typing import Union, Literal, Optional


LogLevel = Union[Literal[10, 20, 30, 40, 50], int]


class DefaultLogger:
    """Default logger with file and stream handlers."""

    def __init__(
        self,
        level     : LogLevel = logging.INFO,
        directory : Union[str, pathlib.Path] = None,
        name      : Optional[str] = None,
        reinit    : bool = True,
        file_open_mode : str = 'a',
    ):
        self.level = level
        self.directory = pathlib.Path(directory) if directory else None
        self.name = name
        self.reinit = reinit
        self.file_open_mode = 'w' if reinit else file_open_mode

    def setup(self) -> logging.Logger:
        """Setup logger with default configuration."""
        # Create directory if needed
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)

        logger = logging.getLogger(self.name or __name__)

        # Skip if already configured and reinit=False
        if not self.reinit and logger.handlers:
            return logger

        # Clear existing handlers if reinit=True
        if self.reinit:
            logger.handlers.clear()

        logger.setLevel(self.level)

        # Add handlers
        self.add_handlers(logger)

        return logger

    def add_handlers(self, logger: logging.Logger) -> None:
        """Add default handlers."""
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)5s] %(message)s',
            datefmt='%H:%M:%S'
        )

        # Add file handler
        if self.directory:
            file_handler = logging.FileHandler(
                self.directory / 'logging.log',
                mode=self.file_open_mode,
                encoding='utf8',
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add stream handler
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


class RichLogger(DefaultLogger):
    """Rich logger that extends default logger with RichHandler."""

    def __init__(
        self,
        level     : LogLevel = logging.INFO,
        directory : Union[str, pathlib.Path] = None,
        name      : Optional[str] = None,
        reinit    : bool = True,
        file_open_mode  : str = 'a',
        rich_tracebacks : bool = True,
    ):
        super().__init__(level, directory, name, reinit, file_open_mode)
        self.rich_tracebacks = rich_tracebacks

    def add_handlers(self, logger: logging.Logger) -> None:
        """Add rich console handler and file handler."""
        # Add rich console handler (replaces stream handler)
        rich_handler = RichHandler(rich_tracebacks=self.rich_tracebacks)
        logger.addHandler(rich_handler)

        # Add file handler (same as parent)
        if self.directory:
            formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)5s] %(message)s',
                datefmt='%H:%M:%S'
            )
            file_handler = logging.FileHandler(
                self.directory / 'logging.log',
                mode=self.file_open_mode,
                encoding='utf8',
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


def enable_default_logger(
    level     : LogLevel = logging.DEBUG,
    directory : Union[str, pathlib.Path] = None,
    name      : Optional[str] = None,
    reinit    : bool = True,
    file_open_mode : str = 'a',
) -> logging.Logger:
    """Create default logger."""
    return DefaultLogger(level, directory, name, reinit, file_open_mode).setup()


def enable_rich_logger(
    level     : LogLevel = logging.DEBUG,
    directory : Union[str, pathlib.Path] = None,
    name      : Optional[str] = None,
    reinit    : bool = True,
    file_open_mode  : str = 'a',
    rich_tracebacks : bool = True,
) -> logging.Logger:
    """Create rich logger."""
    return RichLogger(level, directory, name, reinit, file_open_mode, rich_tracebacks).setup()


class CaptureLogHandler(logging.Handler):
    """Custom handler to capture log output"""

    def __init__(self):
        super().__init__()
        self.log_buffer = StringIO()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_buffer.write(msg + '\n')
        except Exception:
            self.handleError(record)

    def get_logs(self):
        return self.log_buffer.getvalue()


class OutputCapture:
    """Context manager to capture stdout, stderr, and logger output"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()
        self.log_handler = None
        self.stdout_ctx = None
        self.stderr_ctx = None

    def __enter__(self):
        # Capture stdout/stderr
        self.stdout_ctx = redirect_stdout(self.stdout_capture)
        self.stderr_ctx = redirect_stderr(self.stderr_capture)
        self.stdout_ctx.__enter__()
        self.stderr_ctx.__enter__()

        # Capture logger if provided
        if self.logger:
            self.log_handler = CaptureLogHandler()
            # Copy formatter from existing handler if available
            if self.logger.handlers:
                self.log_handler.setFormatter(self.logger.handlers[0].formatter)
            self.logger.addHandler(self.log_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout/stderr
        if self.stdout_ctx:
            self.stdout_ctx.__exit__(exc_type, exc_val, exc_tb)
        if self.stderr_ctx:
            self.stderr_ctx.__exit__(exc_type, exc_val, exc_tb)

        # Remove logger handler
        if self.log_handler and self.logger:
            self.logger.removeHandler(self.log_handler)

        return False  # Don't suppress exceptions

    def get_output(self):
        """Get all captured output"""
        return {
            'stdout': self.stdout_capture.getvalue(),
            'stderr': self.stderr_capture.getvalue(),
            'logs': self.log_handler.get_logs() if self.log_handler else ''
        }
