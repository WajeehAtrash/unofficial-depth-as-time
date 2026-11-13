"""Unofficial implementation of "Depth as Time in One-Step Generative Models".

Convention used everywhere in this package: data lives at t = 0, noise at t = 1 and the
probability path is linear, ``x_t = (1 - t) * x_0 + t * eps``.
"""

from depth_as_time.decoding import clean_from_velocity, endpoint_from_velocity
from depth_as_time.mixing import MixingResult, run_mixing_analysis
from depth_as_time.models.base import LayerwiseModel
from depth_as_time.probe import LayerwiseProbe, ProbeResult
from depth_as_time.task import TransportTask
from depth_as_time.trajectory import multi_step_remaining, probe_sampling_trajectory

__all__ = [
    "LayerwiseModel",
    "LayerwiseProbe",
    "MixingResult",
    "ProbeResult",
    "TransportTask",
    "clean_from_velocity",
    "endpoint_from_velocity",
    "multi_step_remaining",
    "probe_sampling_trajectory",
    "run_mixing_analysis",
]

__version__ = "0.1.0"
