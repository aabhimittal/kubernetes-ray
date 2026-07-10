"""Tests for the hierarchical checkpoint manager (no torch required)."""

import asyncio

import pytest

from src.core.checkpoint_manager import (
    CheckpointTier,
    HierarchicalCheckpointManager,
)


def test_should_checkpoint_frequency():
    tier = CheckpointTier("L1", "/dev/shm/test_ckpt", frequency=100, max_keep=5)
    assert tier.should_checkpoint(100)
    assert tier.should_checkpoint(200)
    assert not tier.should_checkpoint(150)


def test_object_storage_uri_does_not_mkdir(tmp_path):
    # Should not raise even though s3:// is not a local path.
    tier = CheckpointTier("L4", "s3://bucket/ckpt", frequency=5000, max_keep=10)
    assert tier.name == "L4"


def test_manager_tiers_configured():
    mgr = HierarchicalCheckpointManager(
        l1_path="/dev/shm/ckpt_l1",
        l2_path="/tmp/ckpt_l2",
        l3_path="/tmp/ckpt_l3",
        l4_path="s3://bucket/ckpt_l4",
    )
    assert set(mgr.tiers.keys()) == {"L1", "L2", "L3", "L4"}
    assert mgr.tiers["L1"].frequency == 100
    assert mgr.tiers["L4"].frequency == 5000


def test_validate_checkpoint():
    mgr = HierarchicalCheckpointManager(
        l1_path="/tmp/ckpt_v1",
        l2_path="/tmp/ckpt_v2",
        l3_path="/tmp/ckpt_v3",
        l4_path="s3://bucket/ckpt_v4",
    )
    good = {"model": {}, "optimizer": {}, "step": 1, "epoch": 0}
    bad = {"model": {}, "step": 1}
    assert mgr._validate_checkpoint(good)
    assert not mgr._validate_checkpoint(bad)


def test_hash_is_deterministic():
    mgr = HierarchicalCheckpointManager(
        l1_path="/tmp/ckpt_h1",
        l2_path="/tmp/ckpt_h2",
        l3_path="/tmp/ckpt_h3",
        l4_path="s3://bucket/ckpt_h4",
    )
    state = {"a": 1, "b": [1, 2, 3]}
    assert mgr._compute_hash(state) == mgr._compute_hash(state)


class _FakeModule:
    """Minimal stand-in for torch.nn.Module / optimizer with state_dict."""

    def __init__(self, payload):
        self._payload = payload

    def state_dict(self):
        return self._payload


def test_save_checkpoint_writes_json_metadata(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")  # skip cleanly if torch missing

    mgr = HierarchicalCheckpointManager(
        l1_path=str(tmp_path / "l1"),
        l2_path=str(tmp_path / "l2"),
        l3_path=str(tmp_path / "l3"),
        l4_path=str(tmp_path / "l4"),
    )
    # Force every tier to checkpoint at step 100.
    for tier in mgr.tiers.values():
        tier.frequency = 100

    model = _FakeModule({"w": torch.zeros(2)})
    optimizer = _FakeModule({"lr": 0.1})

    asyncio.run(
        mgr.save_checkpoint(model, optimizer, step=100, epoch=0, metrics={"loss": 1.0})
    )

    stats = mgr.get_checkpoint_stats()
    assert stats["L1"]["count"] == 1
    assert stats["L1"]["latest_step"] == 100


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
