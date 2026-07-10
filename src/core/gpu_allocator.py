"""
Adaptive GPU Allocator with Workload Profiling

Innovation: Uses online learning to predict optimal GPU allocation
based on historical workload patterns.

Deep Dive: This implements a Thompson Sampling bandit algorithm where
each "arm" represents a GPU configuration (number + placement). We
learn which configurations maximize throughput for different workload
types.

Mathematical Foundation:
    Beta distribution parameters: alpha (successes), beta (failures)
    Posterior: Beta(alpha + s, beta + f) where s = successes, f = failures
    Sampling: theta ~ Beta(alpha, beta)
    Selection: argmax(theta_i)

This naturally balances exploration vs exploitation!
"""

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class WorkloadProfile:
    """Characterizes a training workload."""

    model_size: int  # Parameters
    batch_size: int
    sequence_length: int
    precision: str  # fp32, fp16, bf16
    gradient_checkpointing: bool

    def fingerprint(self) -> str:
        """Create a unique identifier for the workload type."""
        return (
            f"{self.model_size}_{self.batch_size}_"
            f"{self.sequence_length}_{self.precision}"
        )


@dataclass
class AllocationArm:
    """Bandit arm representing a GPU configuration."""

    num_gpus: int
    strategy: str  # PACK, SPREAD, etc.
    alpha: float = 1.0  # Beta distribution parameter
    beta: float = 1.0  # Beta distribution parameter
    samples: int = 0
    total_throughput: float = 0.0

    def sample_theta(self) -> float:
        """Sample from the Beta posterior."""
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, success: bool, throughput: float) -> None:
        """Update the posterior with a new observation."""
        if success:
            self.alpha += throughput / 100.0  # Normalize throughput
        else:
            self.beta += 1.0
        self.samples += 1
        self.total_throughput += throughput

    def mean_throughput(self) -> float:
        """Calculate the mean observed throughput."""
        return self.total_throughput / self.samples if self.samples > 0 else 0.0


class AdaptiveGPUAllocator:
    """
    Adaptive allocator using a multi-armed bandit algorithm.

    Lifecycle:
    1. Receive workload profile
    2. Sample from Beta distributions for each arm
    3. Select arm with highest sample
    4. Allocate GPUs according to selected configuration
    5. Observe throughput
    6. Update posterior

    Over time the allocator converges to optimal configurations for
    each workload type while still exploring alternatives.
    """

    def __init__(self, max_gpus: int = 8, state_file: str = "allocator_state.pkl"):
        self.max_gpus = max_gpus
        self.state_file = state_file
        self.arms: Dict[str, Dict[Tuple[int, str], AllocationArm]] = defaultdict(dict)
        self._initialize_arms()
        self._load_state()

    def _initialize_arms(self) -> None:
        """Create all possible allocation arms."""
        strategies = ["STRICT_PACK", "PACK", "SPREAD", "STRICT_SPREAD"]
        for num_gpus in range(1, self.max_gpus + 1):
            for strategy in strategies:
                if strategy == "STRICT_SPREAD" and num_gpus > 4:
                    continue  # Assume max 4 nodes
                arm = AllocationArm(num_gpus=num_gpus, strategy=strategy)
                self.arms["default"][(num_gpus, strategy)] = arm

    def select_allocation(
        self, workload: WorkloadProfile, available_gpus: int
    ) -> Tuple[int, str]:
        """
        Select optimal GPU allocation using Thompson Sampling.

        The beauty of Thompson Sampling: it automatically balances
        exploitation (choosing known good arms) with exploration
        (trying uncertain arms).
        """
        fingerprint = workload.fingerprint()

        if fingerprint not in self.arms:
            self.arms[fingerprint] = {
                k: AllocationArm(num_gpus=v.num_gpus, strategy=v.strategy)
                for k, v in self.arms["default"].items()
            }

        workload_arms = self.arms[fingerprint]
        feasible_arms = {
            k: v for k, v in workload_arms.items() if v.num_gpus <= available_gpus
        }

        if not feasible_arms:
            return min(available_gpus, self.max_gpus), "STRICT_PACK"

        sampled_thetas = {k: v.sample_theta() for k, v in feasible_arms.items()}
        best_arm = max(sampled_thetas.items(), key=lambda x: x[1])[0]
        num_gpus, strategy = best_arm
        return num_gpus, strategy

    def update_allocation(
        self,
        workload: WorkloadProfile,
        num_gpus: int,
        strategy: str,
        throughput: float,
        success: bool = True,
    ) -> None:
        """Update the allocator with observed performance."""
        fingerprint = workload.fingerprint()
        arm_key = (num_gpus, strategy)

        if fingerprint not in self.arms:
            self.arms[fingerprint] = {
                k: AllocationArm(num_gpus=v.num_gpus, strategy=v.strategy)
                for k, v in self.arms["default"].items()
            }

        if arm_key in self.arms[fingerprint]:
            self.arms[fingerprint][arm_key].update(success, throughput)
            self._save_state()

    def get_best_allocation(
        self, workload: WorkloadProfile, available_gpus: int
    ) -> Tuple[int, str, float]:
        """
        Get the best known allocation (exploitation only).

        Returns: (num_gpus, strategy, expected_throughput)
        """
        fingerprint = workload.fingerprint()
        if fingerprint not in self.arms:
            return 1, "STRICT_PACK", 0.0

        workload_arms = self.arms[fingerprint]
        feasible_arms = {
            k: v
            for k, v in workload_arms.items()
            if v.num_gpus <= available_gpus and v.samples > 0
        }

        if not feasible_arms:
            return 1, "STRICT_PACK", 0.0

        best_arm = max(feasible_arms.items(), key=lambda x: x[1].mean_throughput())
        (num_gpus, strategy), arm = best_arm
        return num_gpus, strategy, arm.mean_throughput()

    def _save_state(self) -> None:
        """Persist allocator state."""
        with open(self.state_file, "wb") as f:
            pickle.dump(dict(self.arms), f)

    def _load_state(self) -> None:
        """Load previous allocator state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "rb") as f:
                self.arms = defaultdict(dict, pickle.load(f))
