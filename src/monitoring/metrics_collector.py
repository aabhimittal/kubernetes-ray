"""
Prometheus Metrics Collector for GPU Training

Exposes training and GPU metrics in the Prometheus text exposition format so
they can be scraped by the Prometheus deployment in `k8s/monitoring.yaml` and
visualized in Grafana.

The collector works with or without the `prometheus_client` library: if it is
installed we use real Gauges/Counters; otherwise we keep an in-memory registry
and render the exposition format ourselves (handy for tests and lightweight
deployments).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest
    PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - prometheus_client optional
    PROM_AVAILABLE = False


@dataclass
class MetricSample:
    """A single labelled metric value."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""


class MetricsCollector:
    """
    Collects and exposes training/GPU metrics.

    Typical metrics:
        - gpu_utilization{gpu="0"}
        - gpu_memory_used_bytes{gpu="0"}
        - training_throughput_samples_per_second
        - training_loss
        - allocated_gpus
    """

    def __init__(self, namespace: str = "ray_gpu"):
        self.namespace = namespace
        self._samples: Dict[Tuple[str, Tuple], MetricSample] = {}
        self._prom_registry = None
        self._prom_gauges: Dict[str, "Gauge"] = {}
        if PROM_AVAILABLE:
            self._prom_registry = CollectorRegistry()

    def _full_name(self, name: str) -> str:
        return f"{self.namespace}_{name}"

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
    ) -> None:
        """Set a gauge metric to a value."""
        labels = labels or {}
        full = self._full_name(name)
        key = (full, tuple(sorted(labels.items())))
        self._samples[key] = MetricSample(full, value, labels, help_text)

        if PROM_AVAILABLE:
            gauge = self._prom_gauges.get(full)
            if gauge is None:
                gauge = Gauge(
                    full,
                    help_text or full,
                    labelnames=list(labels.keys()),
                    registry=self._prom_registry,
                )
                self._prom_gauges[full] = gauge
            (gauge.labels(**labels) if labels else gauge).set(value)

    def record_gpu_stats(self, gpu_id: int, utilization: float, memory_used: int) -> None:
        """Record per-GPU statistics."""
        self.set_gauge(
            "gpu_utilization",
            utilization,
            {"gpu": str(gpu_id)},
            "GPU utilization ratio (0-1)",
        )
        self.set_gauge(
            "gpu_memory_used_bytes",
            float(memory_used),
            {"gpu": str(gpu_id)},
            "GPU memory used in bytes",
        )

    def record_training_stats(
        self, throughput: float, loss: float, allocated_gpus: int
    ) -> None:
        """Record training-loop statistics."""
        self.set_gauge(
            "training_throughput_samples_per_second",
            throughput,
            help_text="Training throughput (samples/s)",
        )
        self.set_gauge("training_loss", loss, help_text="Current training loss")
        self.set_gauge(
            "allocated_gpus", float(allocated_gpus), help_text="GPUs currently allocated"
        )

    def collect(self) -> List[MetricSample]:
        """Return all current metric samples."""
        return list(self._samples.values())

    def render(self) -> str:
        """Render metrics in the Prometheus text exposition format."""
        if PROM_AVAILABLE:
            return generate_latest(self._prom_registry).decode("utf-8")

        lines: List[str] = []
        seen_help = set()
        for sample in self._samples.values():
            if sample.help_text and sample.name not in seen_help:
                lines.append(f"# HELP {sample.name} {sample.help_text}")
                lines.append(f"# TYPE {sample.name} gauge")
                seen_help.add(sample.name)
            if sample.labels:
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in sorted(sample.labels.items())
                )
                lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
            else:
                lines.append(f"{sample.name} {sample.value}")
        return "\n".join(lines) + "\n"
