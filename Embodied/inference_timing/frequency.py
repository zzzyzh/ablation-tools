"""Conversions between inference latency, call frequency, and replanning rate."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Literal


LatencyUnit = Literal["seconds", "milliseconds", "microseconds"]

_SECONDS_PER_UNIT: dict[LatencyUnit, float] = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
}


def _validate_positive_finite(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return result


def _validate_latency_unit(unit: str) -> LatencyUnit:
    if unit not in _SECONDS_PER_UNIT:
        raise ValueError(
            "unit must be 'seconds', 'milliseconds', or 'microseconds', "
            f"got {unit!r}"
        )
    return unit  # type: ignore[return-value]


def _validate_frequency_result(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} is not representable as a positive finite float")
    return value


def _validate_executed_actions(executed_actions_per_inference: int) -> int:
    if (
        not isinstance(executed_actions_per_inference, Integral)
        or isinstance(executed_actions_per_inference, bool)
    ):
        raise TypeError("executed_actions_per_inference must be a positive integer")
    if executed_actions_per_inference <= 0:
        raise ValueError("executed_actions_per_inference must be positive")
    return int(executed_actions_per_inference)


def convert_latency_to_seconds(latency: float, *, unit: LatencyUnit = "seconds") -> float:
    """Convert one positive latency value to seconds."""

    value = _validate_positive_finite(latency, "latency")
    resolved_unit = _validate_latency_unit(unit)
    seconds = value * _SECONDS_PER_UNIT[resolved_unit]
    return _validate_frequency_result(seconds, "latency in seconds")


def convert_latency_to_hz(latency: float, *, unit: LatencyUnit = "seconds") -> float:
    """Convert sequential inference latency to calls per second (Hz)."""

    frequency = 1.0 / convert_latency_to_seconds(latency, unit=unit)
    return _validate_frequency_result(frequency, "frequency_hz")


def convert_hz_to_latency(
    frequency_hz: float,
    *,
    unit: LatencyUnit = "seconds",
) -> float:
    """Convert a positive frequency to its period in the requested unit."""

    frequency = _validate_positive_finite(frequency_hz, "frequency_hz")
    resolved_unit = _validate_latency_unit(unit)
    latency = (1.0 / frequency) / _SECONDS_PER_UNIT[resolved_unit]
    return _validate_frequency_result(latency, "latency")


def convert_control_frequency_to_replanning_hz(
    control_frequency_hz: float,
    *,
    executed_actions_per_inference: int,
) -> float:
    """Convert low-level control rate and executed action chunk to required replanning Hz."""

    control_frequency = _validate_positive_finite(
        control_frequency_hz,
        "control_frequency_hz",
    )
    executed_actions = _validate_executed_actions(executed_actions_per_inference)
    return _validate_frequency_result(
        control_frequency / executed_actions,
        "replanning_hz",
    )


def compute_synchronous_replanning_hz(
    inference_latency: float,
    control_frequency_hz: float,
    *,
    executed_actions_per_inference: int,
    latency_unit: LatencyUnit = "seconds",
) -> float:
    """Compute infer-then-execute replanning Hz when stages do not overlap."""

    inference_seconds = convert_latency_to_seconds(
        inference_latency,
        unit=latency_unit,
    )
    control_frequency = _validate_positive_finite(
        control_frequency_hz,
        "control_frequency_hz",
    )
    executed_actions = _validate_executed_actions(executed_actions_per_inference)
    cycle_seconds = inference_seconds + executed_actions / control_frequency
    return _validate_frequency_result(1.0 / cycle_seconds, "synchronous_replanning_hz")


def compute_overlapped_replanning_capacity_hz(
    inference_frequency_hz: float,
    control_frequency_hz: float,
    *,
    executed_actions_per_inference: int,
) -> float:
    """Compute the capacity bound when inference and action execution fully overlap."""

    inference_frequency = _validate_positive_finite(
        inference_frequency_hz,
        "inference_frequency_hz",
    )
    action_limited_frequency = convert_control_frequency_to_replanning_hz(
        control_frequency_hz,
        executed_actions_per_inference=executed_actions_per_inference,
    )
    return min(inference_frequency, action_limited_frequency)


__all__ = [
    "LatencyUnit",
    "compute_overlapped_replanning_capacity_hz",
    "compute_synchronous_replanning_hz",
    "convert_control_frequency_to_replanning_hz",
    "convert_hz_to_latency",
    "convert_latency_to_hz",
    "convert_latency_to_seconds",
]
