"""Timer utilities for profiling experiment performance."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str, log_level: int = logging.INFO) -> Generator[None, None, None]:
    """Context manager to time a block of code and log the duration.
    
    Usage:
        with timer("Loading model"):
            model = load_model()
    
    Args:
        label: Description of what's being timed.
        log_level: Logging level (default: INFO).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(log_level, "%s took %.2f seconds", label, elapsed)


def timed(label: str | None = None, log_level: int = logging.INFO) -> Callable:
    """Decorator to time function execution and log the duration.
    
    Usage:
        @timed("Computing SVD")
        def compute_svd(matrix):
            ...
    
    Args:
        label: Description (default: function name).
        log_level: Logging level (default: INFO).
    """
    def decorator(func: Callable) -> Callable:
        func_label = label or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.log(log_level, "%s took %.2f seconds", func_label, elapsed)
        
        return wrapper
    return decorator


class Timer:
    """Reusable timer for measuring multiple operations.
    
    Usage:
        t = Timer()
        t.start("load_model")
        model = load_model()
        t.stop("load_model")
        
        t.start("evaluation")
        evaluate(model)
        t.stop("evaluation")
        
        t.report()  # Print all timings
    """
    
    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}
    
    def start(self, label: str) -> None:
        """Start timing an operation."""
        self._starts[label] = time.perf_counter()
    
    def stop(self, label: str) -> float:
        """Stop timing an operation and return elapsed time in seconds."""
        if label not in self._starts:
            raise ValueError(f"Timer '{label}' was never started")
        elapsed = time.perf_counter() - self._starts[label]
        self._timings[label] = elapsed
        del self._starts[label]
        return elapsed
    
    def get(self, label: str) -> float:
        """Get elapsed time for a completed operation."""
        if label not in self._timings:
            raise ValueError(f"No timing recorded for '{label}'")
        return self._timings[label]
    
    def report(self, log_level: int = logging.INFO) -> None:
        """Log all recorded timings."""
        if not self._timings:
            logger.log(log_level, "No timings recorded")
            return
        
        logger.log(log_level, "=== Timing Report ===")
        total = sum(self._timings.values())
        for label, elapsed in sorted(self._timings.items(), key=lambda x: -x[1]):
            percentage = (elapsed / total * 100) if total > 0 else 0
            logger.log(log_level, "  %s: %.2fs (%.1f%%)", label, elapsed, percentage)
        logger.log(log_level, "  TOTAL: %.2fs", total)
    
    def get_timings(self) -> dict[str, float]:
        """Return all timings as a dictionary."""
        return self._timings.copy()
