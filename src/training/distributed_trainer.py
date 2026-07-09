"""
Distributed Training Orchestrator

Ties together the whole stack:
    - RayClusterManager      -> cluster + placement groups
    - AdaptiveGPUAllocator   -> how many GPUs / which strategy
    - QuantumInspiredScheduler -> task-to-GPU assignment
    - ZeroCopyDataPipeline   -> data feeding
    - MixedPrecisionTrainer  -> fp16/bf16 step
    - HierarchicalCheckpointManager -> fault tolerance

The trainer is intentionally framework-light: you pass in `train_step`,
`model`, and `optimizer` and it drives the loop, delegating the hard
infrastructure decisions to the components above.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.core.checkpoint_manager import HierarchicalCheckpointManager
from src.core.gpu_allocator import AdaptiveGPUAllocator, WorkloadProfile
from src.scheduling.quantum_scheduler import (
    QuantumInspiredScheduler,
    SchedulerConfig,
    Task,
)
from src.training.mixed_precision import MixedPrecisionTrainer, PrecisionConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    workload: WorkloadProfile
    max_gpus: int = 8
    available_gpus: int = 8
    epochs: int = 1
    steps_per_epoch: int = 100
    checkpoint_every: int = 100
    compute_capability: tuple = (8, 0)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)


@dataclass
class TrainingResult:
    """Summary of a training run."""

    epochs_completed: int
    steps_completed: int
    final_metrics: Dict[str, float]
    num_gpus: int
    strategy: str
    wall_time_seconds: float


class DistributedTrainer:
    """
    Orchestrates a distributed training run end to end.

    ``train_step`` has the signature ``(model, optimizer, batch, step) ->
    dict[str, float]`` and returns metrics (must include ``loss``). This keeps
    the orchestrator agnostic to model architecture and framework.
    """

    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_manager: Optional[HierarchicalCheckpointManager] = None,
        allocator: Optional[AdaptiveGPUAllocator] = None,
    ):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.allocator = allocator or AdaptiveGPUAllocator(max_gpus=config.max_gpus)
        self.precision_trainer = MixedPrecisionTrainer(
            compute_capability=config.compute_capability, config=config.precision
        )
        self._scheduler: Optional[QuantumInspiredScheduler] = None

    def _plan_allocation(self) -> tuple:
        """Ask the bandit allocator for a (num_gpus, strategy) plan."""
        num_gpus, strategy = self.allocator.select_allocation(
            self.config.workload, self.config.available_gpus
        )
        num_gpus = max(1, num_gpus)
        self._scheduler = QuantumInspiredScheduler(
            num_gpus=num_gpus, config=SchedulerConfig(seed=0)
        )
        logger.info("Allocation plan: %d GPUs, strategy=%s", num_gpus, strategy)
        return num_gpus, strategy

    def _build_schedule(self, num_gpus: int, num_shards: int) -> Dict[str, int]:
        """Assign data-parallel shards to GPUs via the quantum scheduler."""
        tasks: List[Task] = [
            Task(task_id=f"shard_{i}", load=1.0, affinity_group="dp")
            for i in range(num_shards)
        ]
        schedule = self._scheduler.schedule(tasks)
        return schedule.assignment

    def train(
        self,
        model,
        optimizer,
        train_step: Callable,
        data_iterable=None,
    ) -> TrainingResult:
        """
        Run the training loop.

        Args:
            model: The model object (must expose ``state_dict`` for ckpt).
            optimizer: The optimizer (must expose ``state_dict`` for ckpt).
            train_step: Callable returning a metrics dict including ``loss``.
            data_iterable: Optional iterable of batches; when ``None`` a
                synthetic range is used so the loop is runnable anywhere.
        """
        start = time.time()
        num_gpus, strategy = self._plan_allocation()
        self._build_schedule(num_gpus, num_shards=num_gpus)

        step = 0
        metrics: Dict[str, float] = {}
        throughputs: List[float] = []

        for epoch in range(self.config.epochs):
            data = data_iterable if data_iterable is not None else range(
                self.config.steps_per_epoch
            )
            for batch in data:
                step_start = time.time()
                with self.precision_trainer.autocast():
                    metrics = train_step(model, optimizer, batch, step)
                if "loss" not in metrics:
                    raise ValueError("train_step must return a 'loss' metric")

                elapsed = max(time.time() - step_start, 1e-9)
                throughputs.append(1.0 / elapsed)
                step += 1

                if (
                    self.checkpoint_manager is not None
                    and step % self.config.checkpoint_every == 0
                ):
                    asyncio.run(
                        self.checkpoint_manager.save_checkpoint(
                            model, optimizer, step, epoch, metrics
                        )
                    )

        wall_time = time.time() - start
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0.0

        # Feed performance back into the bandit so future runs improve.
        self.allocator.update_allocation(
            self.config.workload, num_gpus, strategy, avg_throughput, success=True
        )

        logger.info(
            "Training complete: %d steps, %.2f steps/s avg", step, avg_throughput
        )
        return TrainingResult(
            epochs_completed=self.config.epochs,
            steps_completed=step,
            final_metrics=metrics,
            num_gpus=num_gpus,
            strategy=strategy,
            wall_time_seconds=wall_time,
        )
