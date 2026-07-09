"""
Ray Cluster Manager with Adaptive GPU Allocation

Key Innovation: Dynamic resource discovery and allocation based on
workload profiling. Uses a feedback loop to optimize GPU utilization.

Analogy: Like a smart parking system that tracks which spaces are
occupied and directs incoming cars to optimal spots based on their
size and parking duration.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import ray
    from ray.util.placement_group import placement_group
    RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - Ray is optional for unit tests
    ray = None
    placement_group = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUProfile:
    """GPU characteristics for intelligent allocation."""

    gpu_id: int
    compute_capability: Tuple[int, int]
    memory_total: int
    memory_available: int
    utilization: float
    temperature: float
    power_usage: float

    def score(self, workload_requirements: Dict) -> float:
        """
        Calculate suitability score for a workload.

        Uses weighted multi-criteria decision making:
        - Memory availability (40%)
        - Utilization (30%)
        - Thermal headroom (20%)
        - Compute capability (10%)
        """
        required = workload_requirements.get("memory", self.memory_total) or self.memory_total
        memory_score = self.memory_available / required
        util_score = 1.0 - self.utilization
        thermal_score = max(0.0, 1.0 - (self.temperature / 90.0))
        compute_score = (self.compute_capability[0] * 10 + self.compute_capability[1]) / 90

        return (
            0.4 * memory_score
            + 0.3 * util_score
            + 0.2 * thermal_score
            + 0.1 * compute_score
        )


class RayClusterManager:
    """
    Manages Ray cluster lifecycle with intelligent GPU allocation.

    Mathematical Foundation:
    Allocation problem as Integer Linear Programming:

        minimize: Sum(cost_i * x_i)
        subject to:
            Sum(gpu_mem_i * x_i) >= required_memory
            Sum(x_i) <= available_gpus
            x_i in {0, 1}

    We use a greedy approximation with a scoring function for
    O(n log n) complexity.
    """

    def __init__(
        self,
        namespace: str = "gpu-training",
        dashboard_host: str = "0.0.0.0",
        dashboard_port: int = 8265,
    ):
        self.namespace = namespace
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.gpu_profiles: Dict[int, GPUProfile] = {}
        self._initialized = False

    def initialize(
        self,
        address: Optional[str] = None,
        runtime_env: Optional[Dict] = None,
    ) -> None:
        """Initialize Ray cluster with GPU discovery."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed. `pip install ray[default]`.")

        if self._initialized:
            logger.warning("Cluster already initialized")
            return

        ray_config = {
            "namespace": self.namespace,
            "dashboard_host": self.dashboard_host,
            "dashboard_port": self.dashboard_port,
            "ignore_reinit_error": True,
        }
        if address:
            ray_config["address"] = address
        if runtime_env:
            ray_config["runtime_env"] = runtime_env

        ray.init(**ray_config)
        self._discover_gpus()
        self._initialized = True
        logger.info("Ray cluster initialized with %d GPUs", len(self.gpu_profiles))

    def _discover_gpus(self) -> None:
        """
        Discover and profile available GPUs.

        Uses NVIDIA Management Library (NVML) for detailed GPU stats.
        Falls back to Ray's resource detection if NVML is unavailable.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # watts
                compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

                self.gpu_profiles[i] = GPUProfile(
                    gpu_id=i,
                    compute_capability=compute_cap,
                    memory_total=memory_info.total,
                    memory_available=memory_info.free,
                    utilization=utilization.gpu / 100.0,
                    temperature=temperature,
                    power_usage=power,
                )
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - broad on purpose, NVML is optional
            logger.warning("pynvml not available, using basic GPU detection")
            available_gpus = 0
            if RAY_AVAILABLE:
                available_gpus = ray.available_resources().get("GPU", 0)
            for i in range(int(available_gpus)):
                self.gpu_profiles[i] = GPUProfile(
                    gpu_id=i,
                    compute_capability=(8, 0),  # Assume Ampere
                    memory_total=40 * 1024 ** 3,  # Assume 40GB
                    memory_available=40 * 1024 ** 3,
                    utilization=0.0,
                    temperature=0.0,
                    power_usage=0.0,
                )

    def allocate_gpus(self, workload_requirements: Dict, num_gpus: int) -> List[int]:
        """
        Intelligently allocate GPUs based on workload requirements.

        Algorithm:
        1. Score all GPUs based on workload requirements
        2. Sort by score (descending)
        3. Select top N GPUs
        """
        if not self.gpu_profiles:
            self._discover_gpus()

        scored_gpus = [
            (gpu_id, profile.score(workload_requirements))
            for gpu_id, profile in self.gpu_profiles.items()
        ]
        scored_gpus.sort(key=lambda x: x[1], reverse=True)

        allocated = [gpu_id for gpu_id, _ in scored_gpus[:num_gpus]]
        logger.info("Allocated GPUs: %s", allocated)
        return allocated

    def create_placement_group(self, num_gpus: int, strategy: str = "STRICT_PACK"):
        """
        Create a Ray placement group for GPU co-location.

        Strategy Options:
        - STRICT_PACK: All on same node (best for high inter-GPU comms)
        - PACK: Prefer same node
        - SPREAD: Distribute across nodes (better fault tolerance)
        - STRICT_SPREAD: Force different nodes
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed. `pip install ray[default]`.")

        bundles = [{"GPU": 1, "CPU": 4} for _ in range(num_gpus)]
        pg = placement_group(bundles, strategy=strategy)
        ray.get(pg.ready())
        logger.info(
            "Created placement group with %d GPUs using %s strategy", num_gpus, strategy
        )
        return pg

    def get_cluster_stats(self) -> Dict:
        """Get current cluster resource statistics."""
        if not RAY_AVAILABLE:
            return {"gpu_profiles": self.gpu_profiles}

        resources = ray.cluster_resources()
        available = ray.available_resources()
        return {
            "total_gpus": resources.get("GPU", 0),
            "available_gpus": available.get("GPU", 0),
            "total_cpus": resources.get("CPU", 0),
            "available_cpus": available.get("CPU", 0),
            "total_memory": resources.get("memory", 0),
            "available_memory": available.get("memory", 0),
            "nodes": len(ray.nodes()),
            "gpu_profiles": self.gpu_profiles,
        }

    def shutdown(self) -> None:
        """Gracefully shut down the Ray cluster."""
        if self._initialized and RAY_AVAILABLE:
            ray.shutdown()
            self._initialized = False
            logger.info("Ray cluster shutdown complete")
