"""Callable-level inference benchmarking with explicit synchronization."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import torch

from .frequency import convert_latency_to_hz


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


@dataclass(frozen=True)
class InferenceLatencyResult:
    """Raw durations and summary statistics for sequential inference calls."""

    durations_seconds: tuple[float, ...]
    warmup_runs: int
    timed_runs: int
    batch_size: int
    timing_scope: str = "callable"

    def __post_init__(self) -> None:
        durations = tuple(float(duration) for duration in self.durations_seconds)
        if not durations:
            raise ValueError("durations_seconds must not be empty")
        if any(not math.isfinite(duration) or duration <= 0 for duration in durations):
            raise ValueError("durations_seconds must contain positive finite values")
        _validate_nonnegative_integer(self.warmup_runs, "warmup_runs")
        if not isinstance(self.timed_runs, Integral) or isinstance(self.timed_runs, bool):
            raise TypeError("timed_runs must be a positive integer")
        if not isinstance(self.batch_size, Integral) or isinstance(self.batch_size, bool):
            raise TypeError("batch_size must be a positive integer")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.timed_runs != len(durations):
            raise ValueError("timed_runs must equal the number of durations")
        if not isinstance(self.timing_scope, str) or not self.timing_scope.strip():
            raise ValueError("timing_scope must be a non-empty string")
        object.__setattr__(self, "durations_seconds", durations)
        object.__setattr__(self, "timing_scope", self.timing_scope.strip())

    @property
    def mean_latency_seconds(self) -> float:
        return statistics.fmean(self.durations_seconds)

    @property
    def mean_latency_milliseconds(self) -> float:
        return self.mean_latency_seconds * 1000.0

    @property
    def median_latency_seconds(self) -> float:
        return statistics.median(self.durations_seconds)

    @property
    def standard_deviation_seconds(self) -> float:
        return statistics.pstdev(self.durations_seconds)

    @property
    def minimum_latency_seconds(self) -> float:
        return min(self.durations_seconds)

    @property
    def maximum_latency_seconds(self) -> float:
        return max(self.durations_seconds)

    @property
    def inference_frequency_hz(self) -> float:
        """Sequential calls per second derived from mean call latency."""

        return convert_latency_to_hz(self.mean_latency_seconds)

    @property
    def samples_per_second(self) -> float:
        """Batch throughput, distinct from sequential call frequency."""

        return self.batch_size * self.inference_frequency_hz

    def percentile_latency_seconds(self, percentile: float) -> float:
        """Linearly interpolate a percentile in ``[0, 100]``."""

        if not isinstance(percentile, Real) or isinstance(percentile, bool):
            raise TypeError("percentile must be a finite number in [0, 100]")
        value = float(percentile)
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("percentile must be finite and in [0, 100]")
        ordered = sorted(self.durations_seconds)
        position = value / 100.0 * (len(ordered) - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return ordered[lower_index]
        fraction = position - lower_index
        return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction

    @property
    def p50_latency_seconds(self) -> float:
        return self.percentile_latency_seconds(50.0)

    @property
    def p90_latency_seconds(self) -> float:
        return self.percentile_latency_seconds(90.0)

    @property
    def p95_latency_seconds(self) -> float:
        return self.percentile_latency_seconds(95.0)

    @property
    def p99_latency_seconds(self) -> float:
        return self.percentile_latency_seconds(99.0)


def create_torch_device_synchronizer(
    device: str | torch.device,
) -> Callable[[], None]:
    """Create a synchronization callback for CPU, CUDA, or supported backends."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cpu":
        return lambda: None
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA synchronization requested but CUDA is unavailable")
        return lambda: torch.cuda.synchronize(resolved_device)
    backend = getattr(torch, resolved_device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise ValueError(
            f"PyTorch backend {resolved_device.type!r} has no synchronize interface; "
            "provide a custom callback"
        )
    return synchronize


def benchmark_inference(
    inference_fn: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    warmup_runs: int = 10,
    timed_runs: int = 100,
    batch_size: int = 1,
    synchronize: Callable[[], None] | None = None,
    use_inference_mode: bool = True,
    timing_scope: str = "callable",
) -> InferenceLatencyResult:
    """Benchmark a callable with warmup and synchronized wall-clock timing.

    The timed boundary contains only ``inference_fn(*args, **kwargs)`` and the
    post-call synchronization. Put preprocessing/postprocessing inside the
    callable only when end-to-end latency is intended. ``timing_scope`` is
    descriptive metadata and never changes that boundary.
    """

    if not callable(inference_fn):
        raise TypeError("inference_fn must be callable")
    warmup_runs = _validate_nonnegative_integer(warmup_runs, "warmup_runs")
    if not isinstance(timed_runs, Integral) or isinstance(timed_runs, bool):
        raise TypeError("timed_runs must be a positive integer")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be positive")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("args must be a sequence")
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping or None")
    if synchronize is not None and not callable(synchronize):
        raise TypeError("synchronize must be callable or None")
    if not isinstance(use_inference_mode, bool):
        raise TypeError("use_inference_mode must be a bool")
    if not isinstance(timing_scope, str) or not timing_scope.strip():
        raise ValueError("timing_scope must be a non-empty string")
    call_kwargs = dict(kwargs or {})
    synchronize_fn = synchronize or (lambda: None)
    context = torch.inference_mode() if use_inference_mode else nullcontext()

    with context:
        for _ in range(warmup_runs):
            inference_fn(*args, **call_kwargs)
        synchronize_fn()

        durations: list[float] = []
        for _ in range(int(timed_runs)):
            synchronize_fn()
            started_ns = time.perf_counter_ns()
            inference_fn(*args, **call_kwargs)
            synchronize_fn()
            duration = (time.perf_counter_ns() - started_ns) * 1e-9
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError("measured inference duration must be positive and finite")
            durations.append(duration)

    return InferenceLatencyResult(
        durations_seconds=tuple(durations),
        warmup_runs=warmup_runs,
        timed_runs=int(timed_runs),
        batch_size=int(batch_size),
        timing_scope=timing_scope,
    )


__all__ = [
    "InferenceLatencyResult",
    "benchmark_inference",
    "create_torch_device_synchronizer",
]
