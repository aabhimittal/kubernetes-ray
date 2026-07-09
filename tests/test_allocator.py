"""Tests for the adaptive GPU allocator and cluster scoring."""

import os
import tempfile

import pytest

from src.core.gpu_allocator import (
    AdaptiveGPUAllocator,
    AllocationArm,
    WorkloadProfile,
)
from src.core.ray_cluster import GPUProfile


def make_workload() -> WorkloadProfile:
    return WorkloadProfile(
        model_size=1_000_000,
        batch_size=32,
        sequence_length=512,
        precision="bf16",
        gradient_checkpointing=False,
    )


def test_workload_fingerprint_is_stable():
    w1 = make_workload()
    w2 = make_workload()
    assert w1.fingerprint() == w2.fingerprint()
    assert "bf16" in w1.fingerprint()


def test_allocation_arm_updates_posterior():
    arm = AllocationArm(num_gpus=2, strategy="STRICT_PACK")
    assert arm.mean_throughput() == 0.0
    arm.update(success=True, throughput=200.0)
    assert arm.samples == 1
    assert arm.mean_throughput() == 200.0
    assert arm.alpha > 1.0  # alpha grew on success


def test_select_allocation_respects_available_gpus():
    alloc = AdaptiveGPUAllocator(max_gpus=8, state_file=_tmp_state())
    workload = make_workload()
    num_gpus, strategy = alloc.select_allocation(workload, available_gpus=2)
    assert 1 <= num_gpus <= 2
    assert strategy in {"STRICT_PACK", "PACK", "SPREAD", "STRICT_SPREAD"}


def test_allocator_converges_to_high_throughput_arm():
    alloc = AdaptiveGPUAllocator(max_gpus=4, state_file=_tmp_state())
    workload = make_workload()

    # Teach it that 4 GPUs STRICT_PACK is great, 1 GPU is poor.
    for _ in range(50):
        alloc.update_allocation(workload, 4, "STRICT_PACK", throughput=400.0)
        alloc.update_allocation(workload, 1, "STRICT_PACK", throughput=10.0)

    num_gpus, strategy, expected = alloc.get_best_allocation(workload, available_gpus=4)
    assert num_gpus == 4
    assert strategy == "STRICT_PACK"
    assert expected > 100.0


def test_allocator_state_persists(tmp_path):
    state_file = str(tmp_path / "state.pkl")
    workload = make_workload()

    alloc = AdaptiveGPUAllocator(max_gpus=4, state_file=state_file)
    alloc.update_allocation(workload, 2, "PACK", throughput=123.0)
    assert os.path.exists(state_file)

    reloaded = AdaptiveGPUAllocator(max_gpus=4, state_file=state_file)
    _, _, tput = reloaded.get_best_allocation(workload, available_gpus=4)
    assert tput > 0.0


def test_gpu_profile_score_prefers_more_memory():
    low_mem = GPUProfile(0, (8, 0), 40 * 1024**3, 5 * 1024**3, 0.5, 60, 200)
    high_mem = GPUProfile(1, (8, 0), 40 * 1024**3, 35 * 1024**3, 0.5, 60, 200)
    req = {"memory": 10 * 1024**3}
    assert high_mem.score(req) > low_mem.score(req)


def _tmp_state() -> str:
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    os.remove(path)  # allocator will recreate on save
    return path


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
