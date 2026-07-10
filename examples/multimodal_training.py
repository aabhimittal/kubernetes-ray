"""
Multimodal Training Example

Demonstrates the quantum-inspired scheduler co-locating tightly-coupled
modality encoders (vision + text + fusion) while balancing load, and the
predictive autoscaler reacting to a growing task queue.

Run:
    python -m examples.multimodal_training
"""

import logging

from src.scheduling.autoscaler import (
    AutoscalerConfig,
    ClusterState,
    PredictiveAutoscaler,
)
from src.scheduling.quantum_scheduler import (
    QuantumInspiredScheduler,
    SchedulerConfig,
    Task,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("multimodal_example")


def main():
    # Three encoders per replica; vision is heavier than text.
    tasks = []
    for replica in range(4):
        group = f"replica_{replica}"
        tasks.append(Task(f"vision_{replica}", load=3.0, affinity_group=group))
        tasks.append(Task(f"text_{replica}", load=1.0, affinity_group=group))
        tasks.append(Task(f"fusion_{replica}", load=1.5, affinity_group=group))

    scheduler = QuantumInspiredScheduler(
        num_gpus=4,
        config=SchedulerConfig(seed=7, affinity_weight=2.0),
    )
    schedule = scheduler.schedule(tasks)

    logger.info("Task -> GPU assignment:")
    for task_id, gpu in sorted(schedule.assignment.items()):
        logger.info("  %-14s -> GPU %d", task_id, gpu)
    logger.info("Per-GPU load: %s (energy=%.3f)", schedule.per_gpu_load, schedule.energy)

    # Autoscaler reacts to a spike in pending multimodal jobs.
    scaler = PredictiveAutoscaler(
        AutoscalerConfig(min_replicas=2, max_replicas=16, scale_up_cooldown=0)
    )
    state = ClusterState(current_replicas=4, gpu_utilization=0.9, pending_tasks=40)
    desired = scaler.desired_replicas(state)
    logger.info("Autoscaler recommends %d replicas (was %d)", desired, state.current_replicas)


if __name__ == "__main__":
    main()
