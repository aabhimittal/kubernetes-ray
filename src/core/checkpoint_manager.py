"""
Hierarchical Checkpoint Manager with Smart Recovery

Innovation: Multi-tiered checkpointing with different frequencies:
- L1 (Memory): Every N steps, ephemeral
- L2 (Local SSD): Every M steps, persists across container restarts
- L3 (Network Storage): Every K steps, persists across node failures
- L4 (Object Storage): Periodic, long-term archival

Analogy: Like a computer's memory hierarchy:
- L1 Cache (fastest, volatile) = In-memory checkpoints
- L2 Cache = Local SSD
- RAM = Network storage
- Hard Drive = Object storage (S3/GCS)

Each level trades off speed vs durability.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional for unit tests
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking."""

    checkpoint_id: str
    step: int
    epoch: int
    timestamp: str
    model_hash: str
    optimizer_hash: str
    metrics: Dict[str, float]
    tier: str  # L1, L2, L3, L4
    size_bytes: int

    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointTier:
    """
    A single tier in the checkpoint hierarchy.

    Each tier has:
    - Storage path
    - Retention policy
    - Async write capability
    """

    def __init__(
        self,
        name: str,
        path: str,
        frequency: int,
        max_keep: int,
        async_write: bool = True,
    ):
        self.name = name
        self.path = Path(path)
        self.frequency = frequency
        self.max_keep = max_keep
        self.async_write = async_write
        self.checkpoints: List[CheckpointMetadata] = []

        # Object-storage style URIs cannot be mkdir'd locally.
        if "://" not in str(path):
            self.path.mkdir(parents=True, exist_ok=True)

    def should_checkpoint(self, step: int) -> bool:
        """Check if this tier should checkpoint at the given step."""
        return step % self.frequency == 0

    async def save_async(self, state_dict: Dict, metadata: CheckpointMetadata) -> None:
        """Asynchronously save a checkpoint."""
        checkpoint_path = self.path / f"checkpoint_{metadata.checkpoint_id}.pt"
        metadata_path = self.path / f"checkpoint_{metadata.checkpoint_id}.json"

        await asyncio.to_thread(torch.save, state_dict, checkpoint_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        self.checkpoints.append(metadata)
        await self._cleanup()
        logger.info("Saved checkpoint to %s: %s", self.name, metadata.checkpoint_id)

    def save_sync(self, state_dict: Dict, metadata: CheckpointMetadata) -> None:
        """Synchronously save a checkpoint."""
        checkpoint_path = self.path / f"checkpoint_{metadata.checkpoint_id}.pt"
        metadata_path = self.path / f"checkpoint_{metadata.checkpoint_id}.json"

        torch.save(state_dict, checkpoint_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        self.checkpoints.append(metadata)
        self._cleanup_sync()
        logger.info("Saved checkpoint to %s: %s", self.name, metadata.checkpoint_id)

    async def _cleanup(self) -> None:
        """Remove old checkpoints based on retention policy."""
        if len(self.checkpoints) <= self.max_keep:
            return

        self.checkpoints.sort(key=lambda x: x.step)
        to_remove = self.checkpoints[: -self.max_keep]
        for ckpt in to_remove:
            checkpoint_path = self.path / f"checkpoint_{ckpt.checkpoint_id}.pt"
            metadata_path = self.path / f"checkpoint_{ckpt.checkpoint_id}.json"
            if checkpoint_path.exists():
                await asyncio.to_thread(os.remove, checkpoint_path)
            if metadata_path.exists():
                await asyncio.to_thread(os.remove, metadata_path)
        self.checkpoints = self.checkpoints[-self.max_keep :]

    def _cleanup_sync(self) -> None:
        """Synchronous cleanup."""
        if len(self.checkpoints) <= self.max_keep:
            return

        self.checkpoints.sort(key=lambda x: x.step)
        to_remove = self.checkpoints[: -self.max_keep]
        for ckpt in to_remove:
            checkpoint_path = self.path / f"checkpoint_{ckpt.checkpoint_id}.pt"
            metadata_path = self.path / f"checkpoint_{ckpt.checkpoint_id}.json"
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
            if metadata_path.exists():
                os.remove(metadata_path)
        self.checkpoints = self.checkpoints[-self.max_keep :]

    def load_latest(self) -> Optional[Dict]:
        """Load the most recent checkpoint from this tier."""
        if not self.checkpoints:
            return None

        latest = max(self.checkpoints, key=lambda x: x.step)
        checkpoint_path = self.path / f"checkpoint_{latest.checkpoint_id}.pt"
        if not checkpoint_path.exists():
            logger.warning("Checkpoint file missing: %s", checkpoint_path)
            return None
        return torch.load(checkpoint_path)


class HierarchicalCheckpointManager:
    """
    Manages multi-tier checkpointing with smart recovery.

    Recovery Strategy:
    1. Try L1 (memory) - fastest
    2. Try L2 (local SSD) - fast, survives container restart
    3. Try L3 (network storage) - medium, survives node failure
    4. Try L4 (object storage) - slow, survives cluster failure
    """

    def __init__(
        self,
        l1_path: str = "/dev/shm/checkpoints",  # Shared memory
        l2_path: str = "/mnt/local-ssd/checkpoints",
        l3_path: str = "/mnt/nfs/checkpoints",
        l4_path: str = "s3://bucket/checkpoints",
    ):
        self.tiers = {
            "L1": CheckpointTier("L1", l1_path, frequency=100, max_keep=5),
            "L2": CheckpointTier("L2", l2_path, frequency=500, max_keep=10),
            "L3": CheckpointTier("L3", l3_path, frequency=1000, max_keep=20),
            "L4": CheckpointTier("L4", l4_path, frequency=5000, max_keep=9999),
        }
        self.current_step = 0
        self.current_epoch = 0

    async def save_checkpoint(
        self,
        model,
        optimizer,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save a checkpoint to the appropriate tiers.

        The async saves happen in parallel, so L4 (slow S3) does not
        block L1/L2 saves.
        """
        self.current_step = step
        self.current_epoch = epoch

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }

        model_hash = self._compute_hash(model.state_dict())
        optimizer_hash = self._compute_hash(optimizer.state_dict())
        checkpoint_id = f"{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_tasks = []
        for tier_name, tier in self.tiers.items():
            if not tier.should_checkpoint(step):
                continue
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                step=step,
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                model_hash=model_hash,
                optimizer_hash=optimizer_hash,
                metrics=metrics,
                tier=tier_name,
                size_bytes=self._estimate_size(state_dict),
            )
            if tier.async_write:
                save_tasks.append(tier.save_async(state_dict, metadata))
            else:
                tier.save_sync(state_dict, metadata)

        if save_tasks:
            await asyncio.gather(*save_tasks)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load a checkpoint using hierarchical fallback (L1 -> L4)."""
        for tier_name in ["L1", "L2", "L3", "L4"]:
            tier = self.tiers[tier_name]
            state_dict = tier.load_latest()
            if state_dict:
                if self._validate_checkpoint(state_dict):
                    logger.info("Loaded checkpoint from %s", tier_name)
                    return state_dict
                logger.warning("Checkpoint validation failed for %s", tier_name)

        logger.warning("No valid checkpoint found in any tier")
        return None

    def _compute_hash(self, state_dict: Dict) -> str:
        """Compute the SHA256 hash of a state dict."""
        bytes_data = pickle.dumps(state_dict)
        return hashlib.sha256(bytes_data).hexdigest()

    def _validate_checkpoint(self, state_dict: Dict) -> bool:
        """Validate checkpoint integrity."""
        required_keys = ["model", "optimizer", "step", "epoch"]
        return all(k in state_dict for k in required_keys)

    def _estimate_size(self, state_dict: Dict) -> int:
        """Estimate the size of a checkpoint in bytes."""
        return len(pickle.dumps(state_dict))

    def get_checkpoint_stats(self) -> Dict:
        """Get statistics about checkpoints across all tiers."""
        stats = {}
        for tier_name, tier in self.tiers.items():
            stats[tier_name] = {
                "count": len(tier.checkpoints),
                "latest_step": max(
                    (c.step for c in tier.checkpoints), default=0
                ),
                "total_size": sum(c.size_bytes for c in tier.checkpoints),
            }
        return stats
