"""
Kubernetes Autoscaling Logic for Ray GPU Clusters

Innovation: A predictive autoscaler that combines a reactive utilization
signal with a simple exponentially-weighted forecast of pending work, so
the cluster scales up *before* the queue backs up and scales down only
after a cool-down to avoid thrashing.

This module computes desired replica counts; the actual scaling is applied
by a Kubernetes HorizontalPodAutoscaler / cluster-autoscaler (see
`k8s/autoscaler.yaml`) or by calling the Kubernetes API from an operator.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AutoscalerConfig:
    """Autoscaler tuning parameters."""

    min_replicas: int = 1
    max_replicas: int = 16
    target_utilization: float = 0.7  # Target GPU utilization (0-1)
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 300.0  # seconds
    forecast_alpha: float = 0.3  # EWMA smoothing for pending-work forecast


@dataclass
class ClusterState:
    """A snapshot of cluster load used to make scaling decisions."""

    current_replicas: int
    gpu_utilization: float  # Aggregate 0-1
    pending_tasks: int
    tasks_per_replica: float = 4.0  # Rough capacity of one worker


class PredictiveAutoscaler:
    """
    Decides the desired number of GPU worker replicas.

    Scaling law (reactive term):
        desired = ceil(current * utilization / target_utilization)

    Predictive term:
        adds capacity for the smoothed forecast of pending tasks.
    """

    def __init__(self, config: Optional[AutoscalerConfig] = None):
        self.config = config or AutoscalerConfig()
        self._pending_ewma: float = 0.0
        self._last_scale_up: float = 0.0
        self._last_scale_down: float = 0.0
        self._initialized = False

    def _clamp(self, replicas: int) -> int:
        return max(self.config.min_replicas, min(self.config.max_replicas, replicas))

    def desired_replicas(self, state: ClusterState, now: Optional[float] = None) -> int:
        """Compute the desired replica count for the given cluster state."""
        now = time.time() if now is None else now

        # Update the pending-work forecast (EWMA).
        if not self._initialized:
            self._pending_ewma = float(state.pending_tasks)
            self._initialized = True
        else:
            a = self.config.forecast_alpha
            self._pending_ewma = a * state.pending_tasks + (1 - a) * self._pending_ewma

        target = max(self.config.target_utilization, 1e-6)

        # Reactive component based on current utilization.
        reactive = math_ceil(state.current_replicas * state.gpu_utilization / target)

        # Predictive component based on the forecast queue depth.
        predictive = math_ceil(self._pending_ewma / max(state.tasks_per_replica, 1e-6))

        raw_desired = max(reactive, predictive, self.config.min_replicas)
        desired = self._clamp(raw_desired)

        # Respect cool-downs to avoid thrashing.
        if desired > state.current_replicas:
            if now - self._last_scale_up < self.config.scale_up_cooldown:
                return state.current_replicas
            self._last_scale_up = now
            logger.info(
                "Scaling UP %d -> %d (util=%.2f, pending~%.1f)",
                state.current_replicas,
                desired,
                state.gpu_utilization,
                self._pending_ewma,
            )
        elif desired < state.current_replicas:
            if now - self._last_scale_down < self.config.scale_down_cooldown:
                return state.current_replicas
            self._last_scale_down = now
            logger.info(
                "Scaling DOWN %d -> %d (util=%.2f, pending~%.1f)",
                state.current_replicas,
                desired,
                state.gpu_utilization,
                self._pending_ewma,
            )

        return desired


def math_ceil(x: float) -> int:
    """Integer ceiling without importing math for a single call site."""
    return int(x) + (1 if x > int(x) else 0)
