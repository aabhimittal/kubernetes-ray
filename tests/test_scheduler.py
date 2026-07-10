"""Tests for the quantum-inspired scheduler and predictive autoscaler."""

import pytest

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


def test_scheduler_assigns_all_tasks():
    scheduler = QuantumInspiredScheduler(num_gpus=4, config=SchedulerConfig(seed=1))
    tasks = [Task(task_id=f"t{i}", load=1.0) for i in range(8)]
    schedule = scheduler.schedule(tasks)
    assert len(schedule.assignment) == 8
    assert all(0 <= gpu < 4 for gpu in schedule.assignment.values())


def test_scheduler_balances_uniform_load():
    scheduler = QuantumInspiredScheduler(num_gpus=4, config=SchedulerConfig(seed=2))
    tasks = [Task(task_id=f"t{i}", load=1.0) for i in range(16)]
    schedule = scheduler.schedule(tasks)
    loads = schedule.per_gpu_load
    # With 16 equal tasks on 4 GPUs, a good schedule is near-perfectly balanced.
    assert max(loads) - min(loads) <= 2.0


def test_scheduler_empty_tasks():
    scheduler = QuantumInspiredScheduler(num_gpus=2)
    schedule = scheduler.schedule([])
    assert schedule.assignment == {}
    assert schedule.energy == 0.0


def test_scheduler_rejects_zero_gpus():
    with pytest.raises(ValueError):
        QuantumInspiredScheduler(num_gpus=0)


def test_scheduler_respects_affinity():
    # Affinity dominates balance here, so groups should co-locate.
    scheduler = QuantumInspiredScheduler(
        num_gpus=4,
        config=SchedulerConfig(seed=3, balance_weight=0.1, affinity_weight=5.0),
    )
    # Two affinity groups that should each stay co-located.
    tasks = [
        Task("a1", 1.0, affinity_group="A"),
        Task("a2", 1.0, affinity_group="A"),
        Task("b1", 1.0, affinity_group="B"),
        Task("b2", 1.0, affinity_group="B"),
    ]
    schedule = scheduler.schedule(tasks)
    a_gpus = {schedule.assignment["a1"], schedule.assignment["a2"]}
    b_gpus = {schedule.assignment["b1"], schedule.assignment["b2"]}
    # Each group should collapse onto a single GPU in a low-energy solution.
    assert len(a_gpus) == 1
    assert len(b_gpus) == 1


def test_autoscaler_scales_up_on_high_util():
    config = AutoscalerConfig(min_replicas=1, max_replicas=16, scale_up_cooldown=0)
    scaler = PredictiveAutoscaler(config)
    state = ClusterState(current_replicas=4, gpu_utilization=0.95, pending_tasks=20)
    desired = scaler.desired_replicas(state, now=1000.0)
    assert desired > 4


def test_autoscaler_respects_max():
    config = AutoscalerConfig(min_replicas=1, max_replicas=8, scale_up_cooldown=0)
    scaler = PredictiveAutoscaler(config)
    state = ClusterState(current_replicas=8, gpu_utilization=1.0, pending_tasks=1000)
    desired = scaler.desired_replicas(state, now=1000.0)
    assert desired <= 8


def test_autoscaler_cooldown_blocks_rapid_scale_up():
    config = AutoscalerConfig(min_replicas=1, max_replicas=16, scale_up_cooldown=60)
    scaler = PredictiveAutoscaler(config)
    state = ClusterState(current_replicas=2, gpu_utilization=0.99, pending_tasks=50)

    first = scaler.desired_replicas(state, now=1000.0)
    assert first > 2
    # A second call inside the cooldown window must not scale again.
    state2 = ClusterState(current_replicas=first, gpu_utilization=0.99, pending_tasks=50)
    second = scaler.desired_replicas(state2, now=1010.0)
    assert second == first


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
