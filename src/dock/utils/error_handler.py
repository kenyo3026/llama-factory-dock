import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Type


@contextmanager
def suppress_errors(
    logger: Optional[logging.Logger] = None,
    warning_exceptions: Tuple[Type[Exception], ...] = (FileNotFoundError, ValueError),
    error_message_prefix: str = "Unexpected error"
):
    """
    Context manager to suppress exceptions by downgrading them to warnings.

    Catches specified exceptions and converts them to warnings instead of raising,
    allowing execution to continue. This "softens" errors by treating them as
    non-fatal warnings. Other exceptions are still logged as errors.

    Args:
        logger: Logger instance to use for warnings/errors. If None, uses default logger.
        warning_exceptions: Tuple of exception types to suppress (downgrade to warnings).
            Defaults to (FileNotFoundError, ValueError).
        error_message_prefix: Prefix for unexpected error messages.
            Defaults to "Unexpected error".

    Usage:
        with suppress_errors(self.logger):
            result = some_function_that_might_fail()

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> with suppress_errors(logger):
        ...     # FileNotFoundError will be suppressed and logged as warning
        ...     raise FileNotFoundError("file not found")
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        yield
    except warning_exceptions as e:
        # Handle multi-line error messages (split and log each line separately)
        for line in str(e).split('\n'):
            if line.strip():
                logger.warning(line)
    except Exception as e:
        logger.error(f"{error_message_prefix}: {str(e)}")

